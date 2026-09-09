#!/usr/bin/env python3
"""Add conservative, low-confidence floor-plane depth to sparse LiDAR maps."""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

from generate_global_lidar_depth import CloudCache, json_objects, load_poses, quat_rot, colorize, label


def fit_floor(points,C,rng,iterations=500,threshold=.035,min_camera_height=.7):
    # Gravity is world +Z. Restrict hypotheses to plausible planes below camera.
    if len(points)>120000:points=points[rng.choice(len(points),120000,replace=False)]
    best=None
    for _ in range(iterations):
        p=points[rng.integers(0,len(points),3)];n=np.cross(p[1]-p[0],p[2]-p[0]);nn=np.linalg.norm(n)
        if nn<1e-8:continue
        n/=nn
        if n[2]<0:n=-n
        if n[2]<.94:continue
        d=-float(n@p[0]);height=-(n[0]*C[0]+n[1]*C[1]+d)/n[2]
        camera_height=C[2]-height
        if not (min_camera_height<camera_height<2.5):continue
        dist=np.abs(points@n+d);count=int(np.count_nonzero(dist<threshold))
        if best is None or count>best[0]:best=(count,n,d,camera_height)
    if best is None:return None
    count,n,d,ch=best
    inlier=np.abs(points@n+d)<threshold
    if count<300:return None
    # Least-squares z=ax+by+c refinement is stable for near-horizontal planes.
    q=points[inlier];A=np.column_stack((q[:,0],q[:,1],np.ones(len(q))));coef=np.linalg.lstsq(A,q[:,2],rcond=None)[0]
    n=np.array([-coef[0],-coef[1],1.]);n/=np.linalg.norm(n);d=-coef[2]/np.linalg.norm(np.array([-coef[0],-coef[1],1.]))
    rms=float(np.sqrt(np.mean((q@n+d)**2)));return n,d,count,rms,ch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--reconstruction',type=Path,required=True)
    p.add_argument('--input-depth',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--radius-m',type=float,default=4.);p.add_argument('--max-sweeps',type=int,default=60)
    p.add_argument('--max-depth',type=float,default=15.);p.add_argument('--min-row-fraction',type=float,default=.55)
    p.add_argument('--floor-confidence',type=float,default=.2);p.add_argument('--frames',type=int,nargs='*')
    p.add_argument('--seed-dilation-px',type=int,default=31,help='grow only from LiDAR-confirmed floor pixels')
    p.add_argument('--min-camera-height-m',type=float,default=.7,
                   help='reject tabletops and other horizontal planes close to the camera')
    p.add_argument('--preview-frames',type=int,nargs='*',default=[50,108,150,250,400,520]);a=p.parse_args()
    for d in ('depth_npy','depth_mm','masks','confidence','floor_masks','previews'):(a.output/d).mkdir(parents=True,exist_ok=True)
    intr=a.reconstruction/'intrinsics_input.txt'
    if not intr.exists():intr=a.reconstruction/'intrinsics_refined.txt'
    if not intr.exists():intr=a.data/'intrinsics.txt'
    K=np.loadtxt(intr,delimiter=',');Ki=np.linalg.inv(K);poses=load_poses(a.reconstruction/'poses_optimized_tum.txt');names=[x.split()[0] for x in open(a.data/'timestamps.txt')][:len(poses)]
    sweeps=[]
    for x in json_objects(a.data/'key_frames.jsonl'):
        pose=x.get('optimized_pose',x['lio_pose']);sweeps.append((np.asarray(pose['translation'],float),quat_rot(pose['quaternion_xyzw']),a.data/str(x['saved_frame_path']).removeprefix('./')))
    centers=np.asarray([x[0] for x in sweeps]);cache=CloudCache();rng=np.random.default_rng(17);selected=set(a.frames) if a.frames else set(range(len(poses)));report=[];added_total=base_total=0
    for frame in sorted(selected):
        name=names[frame];image=cv2.imread(str(a.data/'undistorted'/name));base=np.load(a.input_depth/'depth_npy'/(Path(name).stem+'.npy')).astype(np.float32)
        if image is None:continue
        _,C,R=poses[frame];dist=np.linalg.norm(centers-C,axis=1);idx=np.flatnonzero(dist<=a.radius_m);idx=idx[np.argsort(dist[idx])[:a.max_sweeps]]
        chunks=[]
        for si in idx:
            Cl,Rl,path=sweeps[int(si)];P=cache.get(path);W=P@Rl.T+Cl
            keep=(np.linalg.norm(W[:,:2]-C[:2],axis=1)<a.radius_m)&(W[:,2]<C[2]-.15)&(W[:,2]>C[2]-2.7)
            if keep.any():chunks.append(W[keep])
        fit=fit_floor(np.vstack(chunks),C,rng,min_camera_height=a.min_camera_height_m) if chunks else None;h,w=base.shape;floor=np.zeros_like(base);floor_mask=np.zeros_like(base,bool)
        if fit is not None:
            n,d,inliers,rms,camera_height=fit;yy,xx=np.mgrid[0:h,0:w];pix=np.column_stack((xx.ravel(),yy.ravel(),np.ones(h*w)));rays_c=pix@Ki.T;rays_w=rays_c@R.T
            denom=rays_w@n;numer=-(C@n+d);t=numer/np.where(np.abs(denom)>1e-8,denom,np.nan);X=C+t[:,None]*rays_w;pc=(X-C)@R;depth=pc[:,2].reshape(h,w)
            candidate=np.isfinite(depth)&(depth>.2)&(depth<a.max_depth)&(yy>=int(a.min_row_fraction*h))
            # Existing nearer geometry is an occluder. Dilate it to avoid filling
            # immediately around furniture silhouettes and depth discontinuities.
            existing=base>0
            Xbase=C+base.ravel()[:,None]*rays_w
            floor_seed=existing&(np.abs((Xbase@n+d).reshape(h,w))<.06)
            k=max(3,a.seed_dilation_px|1)
            supported=cv2.dilate(floor_seed.astype(np.uint8),np.ones((k,k),np.uint8))>0
            occluder=existing&(base<depth-.10)
            blocked=cv2.dilate(occluder.astype(np.uint8),np.ones((21,21),np.uint8))>0
            floor_mask=candidate&supported&~existing&~blocked;floor[floor_mask]=depth[floor_mask]
        combined=np.where(base>0,base,floor).astype(np.float32);valid=combined>0;conf_path=a.input_depth/'confidence'/(Path(name).stem+'.png');old=cv2.imread(str(conf_path),cv2.IMREAD_UNCHANGED)
        conf=(old.astype(np.float32)/65535 if old is not None else (base>0).astype(np.float32));conf[floor_mask]=a.floor_confidence
        stem=Path(name).stem;np.save(a.output/'depth_npy'/(stem+'.npy'),combined);mm=np.zeros_like(base,np.uint16);mm[valid]=np.clip(np.rint(combined[valid]*1000),1,65535).astype(np.uint16)
        cv2.imwrite(str(a.output/'depth_mm'/(stem+'.png')),mm);cv2.imwrite(str(a.output/'masks'/(stem+'.png')),valid.astype(np.uint8)*255);cv2.imwrite(str(a.output/'confidence'/(stem+'.png')),np.rint(conf*65535).astype(np.uint16));cv2.imwrite(str(a.output/'floor_masks'/(stem+'.png')),floor_mask.astype(np.uint8)*255)
        added=int(floor_mask.sum());base_count=int((base>0).sum());added_total+=added;base_total+=base_count
        item={'frame':frame,'image':name,'base_pixels':base_count,'floor_pixels':added,'combined_pixels':int(valid.sum()),'plane':None if fit is None else {'normal':fit[0].tolist(),'d':fit[1],'inliers':fit[2],'rms_m':fit[3],'camera_height_m':fit[4]}};report.append(item)
        if frame in a.preview_frames:
            bc=colorize(base,base>0,a.max_depth);cc=colorize(combined,valid,a.max_depth);fm=np.zeros_like(image);fm[floor_mask]=(0,255,255);overlay=image.copy();overlay[floor_mask]=np.rint(.35*image[floor_mask]+.65*fm[floor_mask]).astype(np.uint8)
            panel=np.hstack((label(image,'Original'),label(bc,'LiDAR depth'),label(cc,'+ floor plane'),label(overlay,'Added floor pixels')));cv2.imwrite(str(a.output/'previews'/(f'{frame:06d}_{stem}_floor_comparison.jpg')),panel,[cv2.IMWRITE_JPEG_QUALITY,94])
        if len(report)%25==0:print('frames',len(report),'floor_added',added_total,flush=True)
    summary={'frames':len(report),'base_valid_pixels':base_total,'floor_added_pixels':added_total,'coverage_gain':added_total/max(base_total,1),'floor_confidence':a.floor_confidence,'per_frame':report};(a.output/'floor_depth_report.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='per_frame'},indent=2))

if __name__=='__main__':main()
