#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np

def read_poses(path):
    poses = {}
    with open(path) as f:
        for line in f:
            z=line.split()
            if len(z)!=8: continue
            ts=float(z[0]); t=np.array(z[1:4],float)
            x,y,zz,w=map(float,z[4:8])
            q=np.array([w,x,y,zz]); q/=np.linalg.norm(q)
            poses[round(ts,6)]=(t,q)
    return poses

def qrot(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def load_frames(data, pose_file, limit):
    poses=read_poses(pose_file); frames=[]
    for line in open(data/'timestamps.txt'):
        name,ts=line.split(); ts=float(ts)
        key=round(ts,6); p=data/'undistorted'/name
        if key not in poses or not p.exists(): continue
        t,q=poses[key]; im=cv2.imread(str(p))
        frames.append(dict(name=name,ts=ts,t=t,q=q,R=qrot(q),im=im))
        if limit>0 and len(frames)>=limit: break
    return frames

def skew(t):
    x,y,z=t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)

def fundamental(a,b,K):
    R1,t1=a['R'],a['t']; R2,t2=b['R'],b['t']
    R=R2.T@R1; t=R2.T@(t1-t2)
    Ki=np.linalg.inv(K)
    return Ki.T@skew(t)@R@Ki

def sampson(p,q,F):
    x=np.array([p[0],p[1],1.]); y=np.array([q[0],q[1],1.])
    Fx=F@x; Fty=F.T@y; n=float(y@Fx)
    return abs(n)/math.sqrt(max(1e-12,Fx[0]**2+Fx[1]**2+Fty[0]**2+Fty[1]**2))

def visual_fundamental(points_a,points_b,threshold,min_inliers=20,min_ratio=.25):
    """Robust image-only geometry used when a weak frame distrusts LIO."""
    if len(points_a)<max(8,min_inliers):return None,None
    pa=np.asarray(points_a,np.float32);pb=np.asarray(points_b,np.float32)
    method=getattr(cv2,'USAC_MAGSAC',cv2.FM_RANSAC)
    F,mask=cv2.findFundamentalMat(pa,pb,method,threshold,.999,10000)
    if F is None or np.asarray(F).shape!=(3,3) or mask is None:return None,None
    mask=mask.ravel().astype(bool);count=int(mask.sum())
    if count<min_inliers or count/len(pa)<min_ratio:return None,None
    def cells(p):
        span=np.ptp(p,axis=0);z=(p-p.min(axis=0))/np.maximum(span,1.)
        ij=np.minimum(np.floor(z*[4,3]).astype(int),[3,2])
        return len({tuple(x) for x in ij})
    if cells(pa[mask])<6 or cells(pb[mask])<6:return None,None
    return np.asarray(F,float),mask

class DSU:
    def __init__(self,n):
        self.p=list(range(n));self.frames={}
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b,frame_a,frame_b):
        a,b=self.find(a),self.find(b)
        if a==b:return True
        fa=self.frames.get(a,{frame_a});fb=self.frames.get(b,{frame_b})
        if fa.intersection(fb):return False
        if len(fa)<len(fb):a,b,fa,fb=b,a,fb,fa
        self.p[b]=a;self.frames[a]=fa|fb
        self.frames.pop(b,None);return True

def projection(f,K):
    R=f['R'].T
    return K@np.column_stack((R,-R@f['t']))

def project(f,K,x):
    c=f['R'].T@(x-f['t'])
    if c[2]<=1e-6:return None
    p=K@c
    return p[:2]/p[2]

def find_loop_pairs(frames,max_gap,min_separation,max_distance,max_per_frame,max_view_angle):
    pairs=set()
    for i,a in enumerate(frames):
        candidates=[]
        forward_a=a['R'][:,2]
        for j in range(i+min_separation,len(frames)):
            b=frames[j]
            distance=float(np.linalg.norm(a['t']-b['t']))
            if distance>max_distance:continue
            angle=math.degrees(math.acos(np.clip(np.dot(forward_a,b['R'][:,2]),-1,1)))
            if angle<=max_view_angle:candidates.append((distance,j))
        for _,j in sorted(candidates)[:max_per_frame]:pairs.add((i,j))
    return sorted(pairs)

def frame_support(dsu, offsets, frames, grid_x=8, grid_y=6, min_track_length=4):
    """Conservative pre-triangulation support used to trigger rescue matching."""
    component_sizes={}
    for n in range(offsets[-1]):
        root=dsu.find(n); component_sizes[root]=component_sizes.get(root,0)+1
    support=[]
    for fi,f in enumerate(frames):
        cells=set(); count=0; h,w=f['im'].shape[:2]
        for k,kp in enumerate(f['kp']):
            if component_sizes.get(dsu.find(offsets[fi]+k),0)<min_track_length:continue
            count+=1; u,v=kp.pt
            cells.add((min(grid_x-1,int(u*grid_x/w)),min(grid_y-1,int(v*grid_y/h))))
        support.append((count,len(cells)))
    return support

def weak_recovery_pairs(frames, weak, normal_gap, recovery_gap,
                        spatial_distance, spatial_per_frame, max_view_angle):
    """Pair weak frames more widely in time and against nearby seed poses."""
    pairs=set()
    for i in weak:
        for j in range(max(0,i-recovery_gap),min(len(frames),i+recovery_gap+1)):
            if i!=j and abs(i-j)>normal_gap:pairs.add(tuple(sorted((i,j))))
        candidates=[]; forward=frames[i]['R'][:,2]
        for j,b in enumerate(frames):
            if i==j or abs(i-j)<=normal_gap:continue
            distance=float(np.linalg.norm(frames[i]['t']-b['t']))
            if distance>spatial_distance:continue
            angle=math.degrees(math.acos(np.clip(np.dot(forward,b['R'][:,2]),-1,1)))
            if angle<=max_view_angle:candidates.append((distance,j))
        for _,j in sorted(candidates)[:spatial_per_frame]:pairs.add(tuple(sorted((i,j))))
    return sorted(pairs)

def triangulate(frames,K,max_gap,epi_px,min_parallax,reproj_px,
                loop_min_separation,loop_max_distance,loop_max_per_frame,
                loop_max_view_angle,cache_dir,three_view_min_parallax,
                target_landmarks=50,target_grid_cells=16,recovery_gap=20,
                recovery_spatial_distance=1.0,recovery_spatial_per_frame=8,
                grid_x=8,grid_y=6,grid_min_sift=20,grid_extra_features=40):

    sift=cv2.SIFT_create(nfeatures=6000)
    feature_cache=cache_dir/'features_v5_weak_recovery';feature_cache.mkdir(parents=True,exist_ok=True)
    offsets=[0]
    for f in frames:
        cache=feature_cache/(Path(f['name']).stem+'.npz')
        if cache.exists():
            z=np.load(cache);xy=z['xy'];des=z['des']
            f['core_count']=int(z['core_count'])
            f['kp']=[cv2.KeyPoint(float(x),float(y),1.0) for x,y in xy]
            f['des']=des if len(des) else None
        else:
            g=cv2.cvtColor(f['im'],cv2.COLOR_BGR2GRAY)
            f['kp'],f['des']=sift.detectAndCompute(g,None)
            f['core_count']=len(f['kp'])
            f['core_count']=len(f['kp'])
            mask=np.full(g.shape,255,np.uint8)
            for kp in f['kp']:
                cv2.circle(mask,tuple(np.rint(kp.pt).astype(int)),5,0,-1)
            extra=[]
            gy,gx=grid_y,grid_x
            for yy in range(gy):
                for xx in range(gx):
                    y0,y1=yy*g.shape[0]//gy,(yy+1)*g.shape[0]//gy
                    x0,x1=xx*g.shape[1]//gx,(xx+1)*g.shape[1]//gx
                    core_here=sum(x0<=kp.pt[0]<x1 and y0<=kp.pt[1]<y1 for kp in f['kp'])
                    if core_here>=grid_min_sift:continue
                    corners=cv2.goodFeaturesToTrack(g[y0:y1,x0:x1],grid_extra_features,.003,6,mask=mask[y0:y1,x0:x1],blockSize=5)
                    if corners is None:continue
                    for c in corners[:,0]:
                        extra.append(cv2.KeyPoint(float(c[0]+x0),float(c[1]+y0),5.0))
            if extra:
                kp2,des2=sift.compute(g,extra)
                if des2 is not None:
                    f['kp']=list(f['kp'])+list(kp2)
                    f['des']=des2 if f['des'] is None else np.vstack((f['des'],des2))
            xy=np.array([k.pt for k in f['kp']],np.float32)
            des=f['des'] if f['des'] is not None else np.empty((0,128),np.float32)
            np.savez(cache,xy=xy,des=des,core_count=np.int32(f['core_count']))
        offsets.append(offsets[-1]+len(f['kp']))
    dsu=DSU(offsets[-1]); held=[]; accepted=0; loop_matches=0;conflict_rejected=0;pruned_observations=0;quality_rejected=0
    visual_pairs_attempted=0;visual_pairs_accepted=0;visual_matches_accepted=0
    bf=cv2.BFMatcher(cv2.NORM_L2)
    pair_cache=cache_dir/'pairs_v5_weak_recovery';pair_cache.mkdir(parents=True,exist_ok=True)
    local_pairs=[(i,j) for i in range(len(frames))
                 for j in range(i+1,min(len(frames),i+max_gap+1))]
    loop_pairs=find_loop_pairs(frames,max_gap,loop_min_separation,
        loop_max_distance,loop_max_per_frame,loop_max_view_angle)
    loop_set=set(loop_pairs)
    processed_pairs=set()
    def match_pair(i,j,recovery=False):
        nonlocal accepted,loop_matches,conflict_rejected,visual_pairs_attempted,visual_pairs_accepted,visual_matches_accepted
        if (i,j) in processed_pairs:return
        processed_pairs.add((i,j))
        if frames[i]['des'] is None or frames[j]['des'] is None:return
        cache=pair_cache/('%06d_%06d.npz'%(i,j))
        if cache.exists():
            z=np.load(cache)
            qidx=z['qidx'];tidx=z['tidx'];mutual=z['mutual']
        else:
            reverse={}
            if recovery or (i,j) in loop_set:
                for rp in bf.knnMatch(frames[j]['des'],frames[i]['des'],k=2):
                    if len(rp)>1 and rp[0].distance<.75*rp[1].distance:reverse[rp[0].queryIdx]=rp[0].trainIdx
            qidx=[];tidx=[];mutual=[]
            for pair in bf.knnMatch(frames[i]['des'],frames[j]['des'],k=2):
                if len(pair)<2 or pair[0].distance>=.75*pair[1].distance:continue
                m=pair[0]
                qidx.append(m.queryIdx);tidx.append(m.trainIdx)
                mutual.append(reverse.get(m.trainIdx)==m.queryIdx if recovery or (i,j) in loop_set else True)
            np.savez(cache,qidx=np.asarray(qidx,np.int32),tidx=np.asarray(tidx,np.int32),mutual=np.asarray(mutual,bool))
            qidx=np.asarray(qidx,np.int32);tidx=np.asarray(tidx,np.int32);mutual=np.asarray(mutual,bool)
        F=fundamental(frames[i],frames[j],K)
        # Recovery pairs are descriptor-first.  If their mutual matches form a
        # strong, distributed image-only consensus, that geometry can rescue
        # true matches rejected by an inaccurate LIO relative pose.
        Fv=None
        if recovery:
            eligible=np.flatnonzero(mutual)
            pa=[frames[i]['kp'][int(qidx[k])].pt for k in eligible]
            pb=[frames[j]['kp'][int(tidx[k])].pt for k in eligible]
            visual_pairs_attempted+=1
            Fv,_=visual_fundamental(pa,pb,epi_px)
            visual_pairs_accepted+=int(Fv is not None)
        for qi,ti,mu in zip(qidx,tidx,mutual):
            if (recovery or (i,j) in loop_set) and not mu:continue
            p=frames[i]['kp'][qi].pt;q=frames[j]['kp'][ti].pt
            e=sampson(p,q,F)
            prior_ok=e<=epi_px
            visual_ok=Fv is not None and sampson(p,q,Fv)<=epi_px
            if not prior_ok and not visual_ok:continue
            if not recovery and (i*73856093+j*19349663+int(qi))%10==0:
                if qi<frames[i]['core_count'] and ti<frames[j]['core_count']:held.append((i,j,p,q))
                continue
            if not dsu.union(offsets[i]+int(qi),offsets[j]+int(ti),i,j):
                conflict_rejected+=1;continue
            accepted+=1
            visual_matches_accepted+=int(visual_ok and not prior_ok)
            if (i,j) in loop_set:loop_matches+=1
    for i,j in local_pairs+loop_pairs:match_pair(i,j)
    initial_support=frame_support(dsu,offsets,frames,grid_x,grid_y)
    weak=[i for i,(count,cells) in enumerate(initial_support)
          if count<target_landmarks or cells<target_grid_cells]
    recovery_pairs=weak_recovery_pairs(frames,weak,max_gap,recovery_gap,
        recovery_spatial_distance,recovery_spatial_per_frame,loop_max_view_angle)
    before_recovery=accepted
    for i,j in recovery_pairs:match_pair(i,j,True)
    recovered_support=frame_support(dsu,offsets,frames,grid_x,grid_y)
    # Include the DSU root feature itself. The former parent-pointer test
    # accidentally dropped exactly one observation from every track, turning
    # many true four-view tracks into apparent three-view tracks.
    root_sizes={}
    for n in range(offsets[-1]):
        r=dsu.find(n);root_sizes[r]=root_sizes.get(r,0)+1
    groups={}
    for i,f in enumerate(frames):
        for k,kp in enumerate(f['kp']):
            n=offsets[i]+k; r=dsu.find(n)
            if root_sizes[r]>1:groups.setdefault(r,[]).append((i,k,np.array(kp.pt)))
    points=[];tracks=[];coverage_points=[];coverage_tracks=[]
    for obs in groups.values():
        by_frame={}
        for x in obs:by_frame.setdefault(x[0],x)
        obs=sorted(by_frame.values())
        if len(obs)<3:continue
        a,b=obs[0],obs[-1]
        ca,cb=frames[a[0]]['t'],frames[b[0]]['t']
        ra=frames[a[0]]['R']@np.linalg.inv(K)@np.r_[a[2],1.]
        rb=frames[b[0]]['R']@np.linalg.inv(K)@np.r_[b[2],1.]
        angle=math.degrees(math.acos(np.clip(abs(np.dot(ra/np.linalg.norm(ra),rb/np.linalg.norm(rb))),-1,1)))
        if angle<min_parallax:continue
        A=[]
        for fi,ki,uv in obs:
            P=projection(frames[fi],K);u,v=uv
            A.extend((u*P[2]-P[0],v*P[2]-P[1]))
        _,_,vh=np.linalg.svd(np.asarray(A),full_matrices=False)
        Xh=vh[-1]
        if abs(Xh[3])<1e-9:continue
        X=Xh[:3]/Xh[3];errors=[]
        for fi,ki,uv in obs:
            pp=project(frames[fi],K,X)
            errors.append(np.inf if pp is None else float(np.linalg.norm(pp-uv)))
        keep=[k for k,e in enumerate(errors) if np.isfinite(e) and e<=2*reproj_px]
        pruned_observations+=len(obs)-len(keep)
        obs=[obs[k] for k in keep]
        if len(obs)<3:continue
        if len(keep)<len(errors):
            A=[]
            for fi,ki,uv in obs:
                P=projection(frames[fi],K);u,v=uv
                A.extend((u*P[2]-P[0],v*P[2]-P[1]))
            _,_,vh=np.linalg.svd(np.asarray(A),full_matrices=False)
            Xh=vh[-1]
            if abs(Xh[3])<1e-9:continue
            X=Xh[:3]/Xh[3]
        errors=[]
        for fi,ki,uv in obs:
            pp=project(frames[fi],K,X)
            errors.append(np.inf if pp is None else float(np.linalg.norm(pp-uv)))
        if not np.all(np.isfinite(errors)) or np.median(errors)>reproj_px or np.percentile(errors,90)>2*reproj_px:continue
        is_core=all(ki<frames[fi]['core_count'] for fi,ki,uv in obs)
        if is_core and len(obs)==3:
            rays=[X-frames[fi]['t'] for fi,ki,uv in obs]
            rays=[r/(np.linalg.norm(r)+1e-12) for r in rays]
            max_angle=max(math.degrees(math.acos(np.clip(np.dot(rays[a],rays[b]),-1,1))) for a in range(3) for b in range(a+1,3))
            if max_angle<three_view_min_parallax:
                quality_rejected+=1
                continue
        a=obs[0]
        u,v=np.rint(a[2]).astype(int); im=frames[a[0]]['im']
        u=np.clip(u,0,im.shape[1]-1);v=np.clip(v,0,im.shape[0]-1)
        rgb=im[v,u,::-1]
        item=(X,rgb,np.median(errors))
        # Grid-balanced features are admitted to BA only after stronger
        # multi-view validation. They are no longer merely diagnostic points.
        grid_strong=(not is_core and len(obs)>=4 and np.median(errors)<=reproj_px
                     and np.percentile(errors,90)<=1.5*reproj_px)
        if is_core or grid_strong:
            points.append(item);tracks.append(obs)
        else:
            coverage_points.append(item);coverage_tracks.append(obs)
    final_support=[[0,set()] for _ in frames]
    for (X,c,e),tr in zip(points,tracks):
        if len(tr)<4 or e>reproj_px:continue
        for fi,ki,uv in tr:
            h,w=frames[fi]['im'].shape[:2];u,v=uv
            final_support[fi][0]+=1
            final_support[fi][1].add((min(grid_x-1,int(u*grid_x/w)),min(grid_y-1,int(v*grid_y/h))))
    final_support=[(count,len(cells)) for count,cells in final_support]
    held_err=[sampson(p,q,fundamental(frames[i],frames[j],K)) for i,j,p,q in held]
    loop_landmarks=sum(any(tr[k+1][0]-tr[k][0]>max_gap for k in range(len(tr)-1)) for tr in tracks)
    return points,tracks,coverage_points,held,dict(features=offsets[-1],accepted_matches=accepted,heldout_matches=len(held),
        conflicting_track_merges_rejected=conflict_rejected,
        pruned_track_observations=pruned_observations,
        moderate_quality_tracks_rejected=quality_rejected,
        coverage_landmarks=len(coverage_points),coverage_observations=sum(map(len,coverage_tracks)),
        weak_images_detected=len(weak),recovery_pairs=len(recovery_pairs),
        recovery_accepted_matches=accepted-before_recovery,
        visual_recovery_pairs_attempted=visual_pairs_attempted,
        visual_recovery_pairs_accepted=visual_pairs_accepted,
        visual_only_recovery_matches=visual_matches_accepted,
        images_meeting_target_before=sum(c>=target_landmarks and g>=target_grid_cells for c,g in initial_support),
        images_meeting_target_after_matching=sum(c>=target_landmarks and g>=target_grid_cells for c,g in recovered_support),
        images_meeting_strong_target_final=sum(c>=target_landmarks and g>=target_grid_cells for c,g in final_support),
        target_landmarks_per_image=target_landmarks,target_occupied_grid_cells=target_grid_cells,
        per_image_support_before=[{'landmarks':c,'occupied_grid_cells':g} for c,g in initial_support],
        per_image_support_after_matching=[{'landmarks':c,'occupied_grid_cells':g} for c,g in recovered_support],
        per_image_strong_support_final=[{'landmarks':c,'occupied_grid_cells':g} for c,g in final_support],
        loop_candidates=len(loop_pairs),accepted_loop_matches=loop_matches,loop_landmarks=loop_landmarks,
        heldout_epipolar_median_px=float(np.median(held_err)) if held_err else None,
        heldout_epipolar_p90_px=float(np.percentile(held_err,90)) if held_err else None)

def save(out,points,tracks,coverage_points,held,stats,min_points=100):
    out.mkdir(parents=True,exist_ok=True)
    with open(out/'sparse_points.ply','w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n'%len(points))
        f.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float error\nend_header\n')
        for x,c,e in points:f.write('%g %g %g %d %d %d %g\n'%(*x,*c,e))
    with open(out/'coverage_points.ply','w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n'%len(coverage_points))
        f.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float error\nend_header\n')
        for x,c,e in coverage_points:f.write('%g %g %g %d %d %d %g\n'%(*x,*c,e))
    with open(out/'tracks.csv','w') as f:
        f.write('track,frame,u,v\n')
        for i,tr in enumerate(tracks):
            for o in tr:f.write('%d,%d,%.9g,%.9g\n'%(i,o[0],o[2][0],o[2][1]))
    with open(out/'heldout.csv','w') as f:
        f.write('frame1,frame2,u1,v1,u2,v2\n')
        for i,j,p,q in held:f.write('%d,%d,%.9g,%.9g,%.9g,%.9g\n'%(i,j,p[0],p[1],q[0],q[1]))
    np.savez_compressed(out/'tracks.npz',
        xyz=np.array([p[0] for p in points]),rgb=np.array([p[1] for p in points]),
        frame=np.array([o[0] for tr in tracks for o in tr],np.int32),
        uv=np.array([o[2] for tr in tracks for o in tr]),
        track=np.array([i for i,tr in enumerate(tracks) for o in tr],np.int32))
    errors=[p[2] for p in points]
    stats.update(triangulated_points=len(points),observations=sum(map(len,tracks)),
                 triangulation_reprojection_median_px=float(np.median(errors)) if errors else None,
                 triangulation_reprojection_p90_px=float(np.percentile(errors,90)) if errors else None)
    with open(out/'triangulation_metrics.json','w') as f:json.dump(stats,f,indent=2)
    print(json.dumps(stats,indent=2))
    if len(points)<min_points:raise SystemExit(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--output',type=Path,default=Path('output/lio_camera_pose'));p.add_argument('--pose-file',type=Path);p.add_argument('--intrinsics',type=Path);p.add_argument('--max-images',type=int,default=40);p.add_argument('--max-pair-gap',type=int,default=6);p.add_argument('--epipolar-px',type=float,default=4);p.add_argument('--min-parallax-deg',type=float,default=.5);p.add_argument('--three-view-min-parallax-deg',type=float,default=.75);p.add_argument('--reprojection-px',type=float,default=4);p.add_argument('--loop-min-separation',type=int,default=20);p.add_argument('--loop-max-distance',type=float,default=.5);p.add_argument('--loop-max-per-frame',type=int,default=3);p.add_argument('--loop-max-view-angle',type=float,default=60);p.add_argument('--min-points',type=int,default=100)
    p.add_argument('--target-landmarks-per-image',type=int,default=50)
    p.add_argument('--target-grid-cells',type=int,default=16,help='target occupied cells in the 8x6 image grid')
    p.add_argument('--weak-recovery-gap',type=int,default=20)
    p.add_argument('--weak-spatial-distance',type=float,default=1.0)
    p.add_argument('--weak-spatial-pairs',type=int,default=8)
    p.add_argument('--grid-min-sift',type=int,default=20)
    p.add_argument('--grid-extra-features',type=int,default=40)
    a=p.parse_args()
    intrinsics=a.intrinsics or a.data/'intrinsics.txt'
    # A rerun can use the last BA solution for spatial-neighbor discovery and
    # epipolar gating; a clean run naturally falls back to the LIO prior.
    previous_optimized=a.output/'poses_optimized_tum.txt'
    pose_file=a.pose_file or (previous_optimized if previous_optimized.exists()
                             else a.output/'poses_lio_prior_tum.txt')
    K=np.loadtxt(intrinsics,delimiter=',')
    frames=load_frames(a.data,pose_file,a.max_images)
    pts,tr,cpts,held,st=triangulate(frames,K,a.max_pair_gap,a.epipolar_px,a.min_parallax_deg,a.reprojection_px,a.loop_min_separation,a.loop_max_distance,a.loop_max_per_frame,a.loop_max_view_angle,a.output/'cache',a.three_view_min_parallax_deg,
        a.target_landmarks_per_image,a.target_grid_cells,a.weak_recovery_gap,
        a.weak_spatial_distance,a.weak_spatial_pairs,8,6,a.grid_min_sift,a.grid_extra_features)
    st.update(pose_file=str(pose_file),intrinsics_file=str(intrinsics),three_view_min_parallax_deg=a.three_view_min_parallax_deg)
    st.update(images=len(frames),time_offset_seconds=-.4)
    save(a.output,pts,tr,cpts,held,st,a.min_points)
if __name__=='__main__':main()
