#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import cv2
import numpy as np

def poses(path):
    out={}
    for line in open(path):
        z=line.split()
        if len(z)!=8:continue
        ts=float(z[0]);t=np.array(z[1:4],np.float64)
        x,y,zz,w=map(float,z[4:8])
        R=np.array([[1-2*(y*y+zz*zz),2*(x*y-zz*w),2*(x*zz+y*w)],
                    [2*(x*y+zz*w),1-2*(x*x+zz*zz),2*(y*zz-x*w)],
                    [2*(x*zz-y*w),2*(y*zz+x*w),1-2*(x*x+y*y)]])
        out[round(ts,6)]=(R,t)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--poses',type=Path,default=Path('output/lio_camera_pose_incremental/poses_optimized_tum.txt'))
    p.add_argument('--output',type=Path,default=Path('output/lio_camera_pose_incremental/dense_fused.ply'))
    p.add_argument('--text-output',type=Path,help='Optional x y z r g b frame_support text export')
    p.add_argument('--voxel',type=float,default=.02)
    p.add_argument('--stride',type=int,default=3)
    p.add_argument('--min-frame-support',type=int,default=2)
    p.add_argument('--max-depth',type=float,default=30)
    a=p.parse_args()
    K=np.loadtxt(a.data/'intrinsics.txt',delimiter=',');fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    ps=poses(a.poses); acc={}; raw=0;used_frames=0
    lines=[x.split() for x in open(a.data/'timestamps.txt')]
    for index,(name,stamp) in enumerate(lines):
        key=round(float(stamp),6)
        if key not in ps:continue
        image=cv2.imread(str(a.data/'undistorted'/name),cv2.IMREAD_COLOR)
        depth_path=a.data/'depth_maps'/(name+'_depth.tiff')
        depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:continue
        ys=np.arange(0,depth.shape[0],a.stride);xs=np.arange(0,depth.shape[1],a.stride)
        u,v=np.meshgrid(xs,ys);d=depth[v,u]
        valid=np.isfinite(d)&(d>.1)&(d<a.max_depth)
        u=u[valid].astype(np.float64);v=v[valid].astype(np.float64);d=d[valid].astype(np.float64)
        if not len(d):continue
        pc=np.column_stack(((u-cx)*d/fx,(v-cy)*d/fy,d))
        R,t=ps[key];pw=pc@R.T+t
        rgb=image[v.astype(int),u.astype(int),::-1].astype(np.float64)
        vox=np.floor(pw/a.voxel).astype(np.int64)
        uniq,inv=np.unique(vox,axis=0,return_inverse=True)
        xyz_sum=np.zeros((len(uniq),3));rgb_sum=np.zeros((len(uniq),3));cnt=np.zeros(len(uniq),np.int64)
        np.add.at(xyz_sum,inv,pw);np.add.at(rgb_sum,inv,rgb);np.add.at(cnt,inv,1)
        xyz=xyz_sum/cnt[:,None];col=rgb_sum/cnt[:,None]
        for q,x,c,n in zip(map(tuple,uniq),xyz,col,cnt):
            if q in acc:
                z=acc[q];z[0]+=x;z[1]+=c;z[2]+=1;z[3]+=int(n)
            else:acc[q]=[x.copy(),c.copy(),1,int(n)]
        raw+=len(d);used_frames+=1
        if (index+1)%50==0:print('frames',index+1,'raw',raw,'voxels',len(acc),flush=True)
    selected=[z for z in acc.values() if z[2]>=a.min_frame_support]
    xyz=np.array([z[0]/z[2] for z in selected],np.float64)
    rgb=np.clip(np.array([z[1]/z[2] for z in selected]),0,255).astype(np.uint8)
    support=np.array([z[2] for z in selected],np.int32)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n'%len(xyz))
        f.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty ushort frame_support\nend_header\n')
        for x,c,s in zip(xyz,rgb,support):f.write('%g %g %g %d %d %d %d\n'%(*x,*c,s))
    if a.text_output:
        a.text_output.parent.mkdir(parents=True,exist_ok=True)
        with open(a.text_output,'w') as f:
            f.write('# x y z red green blue frame_support\n')
            for x,c,s in zip(xyz,rgb,support):
                f.write('%.9g %.9g %.9g %d %d %d %d\n'%(*x,*c,s))
    metrics={'pass':bool(len(xyz)>=100000),'images_used':used_frames,'raw_depth_samples':raw,
      'voxel_size_m':a.voxel,'min_frame_support':a.min_frame_support,'dense_points':len(xyz),
      'median_frame_support':float(np.median(support)) if len(support) else 0,
      'p90_frame_support':float(np.percentile(support,90)) if len(support) else 0,
      'text_output':str(a.text_output) if a.text_output else None,
      'source':'LiDAR-derived depth backprojected with final optimized camera poses and cross-frame voxel fusion'}
    with open(a.output.with_suffix('.json'),'w') as f:json.dump(metrics,f,indent=2)
    print(json.dumps(metrics,indent=2))
    raise SystemExit(0 if metrics['pass'] else 2)
if __name__=='__main__':main()
