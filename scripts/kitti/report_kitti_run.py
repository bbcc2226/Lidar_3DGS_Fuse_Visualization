#!/usr/bin/env python3
"""Report exported BA quality and per-image spatial support without rerunning BA."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw


def stats(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return dict(count=len(a), rmse=float(np.sqrt(np.mean(a*a))), median=float(np.median(a)),
                p90=float(np.percentile(a,90)), p95=float(np.percentile(a,95)), max=float(a.max())) if len(a) else {}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reconstruction',type=Path,required=True)
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--image-directory',type=Path,help='Source images for reports that also include reconstruction-only context frames')
    args=ap.parse_args()
    root=args.reconstruction; report=args.dataset/'report'; report.mkdir(exist_ok=True)
    overlays=report/'frames'; overlays.mkdir(exist_ok=True)
    meta=json.loads((args.prepared/'preparation_summary.json').read_text())
    metrics=json.loads((root/'evaluation.json').read_text())
    names=[s.split()[0] for s in (args.prepared/'timestamps.txt').read_text().splitlines() if s and not s.startswith('#')]
    ids=np.array([int(Path(s).stem) for s in names]); n=len(names)
    prior=np.loadtxt(root/'poses_lio_prior_tum.txt'); final=np.loadtxt(root/'poses_optimized_tum.txt')
    assert len(prior)==len(final)==n
    assert np.allclose(prior[:,0],final[:,0],rtol=0,atol=1e-5)
    rp=Rotation.from_quat(prior[:,4:8]).as_matrix(); rf=Rotation.from_quat(final[:,4:8]).as_matrix()
    shift=np.linalg.norm(final[:,1:4]-prior[:,1:4],axis=1)
    angle=Rotation.from_matrix(np.einsum('nji,njk->nik',rp,rf)).magnitude()*180/np.pi
    k=np.loadtxt(args.prepared/'intrinsics.txt',delimiter=','); ki=np.linalg.inv(k)
    points={}; lengths={}
    for line in (root/'sparse/0/points3D.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        s=line.split(); idx=int(s[0])-1
        points[idx]=[float(x) for x in s[1:4]]; lengths[idx]=(len(s)-8)//2
    obs=np.genfromtxt(root/'observations.csv',delimiter=',',names=True)
    ci=obs['camera_id'].astype(int); li=obs['landmark_id'].astype(int)
    xyz=np.array([points[i] for i in li]); uv=np.column_stack((obs['u'],obs['v']))
    def errors(poses,rot):
        local=np.einsum('nji,nj->ni',rot[ci],xyz-poses[ci,1:4])
        pixels=local@k.T
        err=np.linalg.norm(pixels[:,:2]/pixels[:,2:3]-uv,axis=1)
        err[local[:,2]<=0]=np.nan
        return err
    before=errors(prior,rp); after=errors(final,rf)
    # A report definition, not an uncertainty estimate: long tracks with low final error.
    strong=(np.array([lengths[i] for i in li])>=4)&(after<=1.0)
    width,height=meta['image_dimensions']; grids=np.zeros((n,6,8),dtype=int)
    rows=[]
    for i,name in enumerate(names):
        mask=ci==i; sm=mask&strong
        x=np.clip((uv[sm,0]/width*8).astype(int),0,7); y=np.clip((uv[sm,1]/height*6).astype(int),0,5)
        np.add.at(grids[i],(y,x),1)
        row=dict(image_id=int(ids[i]),filename=name,landmarks=int(mask.sum()),strong_landmarks=int(sm.sum()),
                 strong_occupied_cells=int(np.count_nonzero(grids[i])),grid_cells=48,
                 strong_coverage_percent=100*np.count_nonzero(grids[i])/48,
                 final_rmse_px=stats(after[mask]).get('rmse',float('nan')),
                 final_p90_px=stats(after[mask]).get('p90',float('nan')),
                 prior_pose_final_points_rmse_px=stats(before[mask]).get('rmse',float('nan')),
                 translation_change_m=float(shift[i]),rotation_change_deg=float(angle[i]))
        row['weak_support']=row['strong_landmarks']<50 or row['strong_occupied_cells']<16
        rows.append(row)
        im=Image.open((args.image_directory or args.dataset/'images')/name).convert('RGB'); draw=ImageDraw.Draw(im)
        for gx in range(1,8): draw.line((gx*width/8,0,gx*width/8,height),fill=(150,150,150),width=1)
        for gy in range(1,6): draw.line((0,gy*height/6,width,gy*height/6),fill=(150,150,150),width=1)
        for (u,v),is_strong in zip(uv[mask],strong[mask]):
            color=(40,255,80) if is_strong else (255,165,35)
            draw.ellipse((u-2,v-2,u+2,v+2),fill=color)
        draw.rectangle((0,0,width,22),fill=(0,0,0))
        draw.text((5,5),f'Image {ids[i]} | strong {sm.sum()} / all {mask.sum()} | strong cells {row["strong_occupied_cells"]}/48 | green=strong orange=other',fill='white')
        im.save(overlays/(Path(name).stem+'.jpg'),quality=85)
    with (report/'per_frame.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (report/'per_frame_grid.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['image_id']+[f'row{r}_col{c}' for r in range(6) for c in range(8)])
        writer.writerows([[int(ids[i])]+grids[i].ravel().tolist() for i in range(n)])
    held=np.genfromtxt(root/'heldout.csv',delimiter=',',names=True)
    def epipolar(poses,rot):
        a=held['frame1'].astype(int); b=held['frame2'].astype(int)
        rel=np.einsum('nji,njk->nik',rot[b],rot[a])
        t=np.einsum('nji,nj->ni',rot[b],poses[a,1:4]-poses[b,1:4])
        cross=np.zeros((len(t),3,3)); cross[:,0,1]=-t[:,2];cross[:,0,2]=t[:,1];cross[:,1,0]=t[:,2];cross[:,1,2]=-t[:,0];cross[:,2,0]=-t[:,1];cross[:,2,1]=t[:,0]
        f=ki.T@cross@rel@ki
        x=np.column_stack((held['u1'],held['v1'],np.ones(len(t))))
        y=np.column_stack((held['u2'],held['v2'],np.ones(len(t))))
        fx=np.einsum('nij,nj->ni',f,x); fy=np.einsum('nji,nj->ni',f,y)
        denom=np.sum(fx[:,:2]**2+fy[:,:2]**2,axis=1)
        return np.abs(np.sum(y*fx,axis=1))/np.sqrt(np.maximum(denom,1e-30))
    hp=epipolar(prior,rp); hf=epipolar(final,rf)
    summary=dict(pipeline=metrics,prior_pose_final_structure_reprojection=stats(before),final_reprojection=stats(after),
                 heldout_prior_sampson_px=stats(hp),heldout_final_sampson_px=stats(hf),
                 translation_change_m=stats(shift),rotation_change_deg=stats(angle),
                 strong_definition='Track observed in >=4 exported images and this observation has final reprojection error <=1 px.',
                 weak_definition='Fewer than 50 strong landmarks OR fewer than 16 occupied cells in an 8x6 grid; diagnostic heuristic.',
                 weak_frames=[r['image_id'] for r in rows if r['weak_support']])
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(4,1,figsize=(15,12),sharex=True)
    axes[0].plot(ids,[r['landmarks'] for r in rows],label='All landmarks');axes[0].plot(ids,[r['strong_landmarks'] for r in rows],label='Strong landmarks');axes[0].legend();axes[0].set_ylabel('Count')
    axes[1].plot(ids,[r['strong_coverage_percent'] for r in rows]);axes[1].axhline(100*16/48,color='orange',linestyle='--');axes[1].set_ylabel('Strong grid coverage (%)')
    axes[2].plot(ids,[r['final_rmse_px'] for r in rows]);axes[2].set_ylabel('Final RMSE (px)')
    axes[3].plot(ids,shift,label='Translation (m)');axes[3].plot(ids,angle,label='Rotation (deg)');axes[3].legend();axes[3].set_xlabel('Original image ID')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(report/'per_frame_quality.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,7));mesh=ax.imshow(np.log1p(grids.reshape(n,48).T),aspect='auto',origin='lower',extent=[ids[0]-.5,ids[-1]+.5,-.5,47.5]);ax.set_xlabel('Original image ID');ax.set_ylabel('Grid cell: row × 8 + column');fig.colorbar(mesh,label='log(1 + strong landmark count)');fig.tight_layout();fig.savefig(report/'spatial_distribution.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for values,label in [(hp,'LiDAR prior'),(hf,'Refined camera poses')]:
        ordered=np.sort(values[np.isfinite(values)])
        axes[0].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),label=label)
    axes[0].set_xscale('symlog',linthresh=.1);axes[0].set_xlabel('Held-out square-root Sampson error (px)');axes[0].set_ylabel('Cumulative fraction');axes[0].legend()
    axes[1].hist(after[np.isfinite(after)],bins=60,color='steelblue');axes[1].set_xlabel('Final reprojection error (px)');axes[1].set_ylabel('Retained observations')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(report/'ba_performance.png',dpi=150);plt.close(fig)
    weakest=sorted(rows,key=lambda r:(r['strong_occupied_cells'],r['strong_landmarks']))[:20]
    largest_shift=rows[int(np.argmax(shift))]
    text=f'''# KITTI {ids[0]}–{ids[-1]}: BA and landmark support report

Processed {n} images using LiDAR `optimized_pose` as the immutable interpolated camera-pose prior. Camera–LiDAR time offset: {meta['camera_time_offset_seconds']} s. COLMAP registered {metrics['sparse_export_images']} images; excluded {metrics['sparse_excluded_images']}.

## Findings to review

The largest translation change is at image {largest_shift['image_id']}: {largest_shift['translation_change_m']:.3f} m, with {largest_shift['strong_landmarks']} strong landmarks in {largest_shift['strong_occupied_cells']}/48 grid cells. Large changes with limited spatial support deserve inspection even when final reprojection error is low.

## BA performance

- Final landmarks: {metrics['landmarks']:,}; observations: {metrics['observations']:,}.
- Final reprojection RMSE: {metrics['reprojection_rmse_px']:.4f} px; median: {metrics['reprojection_median_px']:.4f} px; P90: {metrics['reprojection_p90_px']:.4f} px; P95: {metrics['reprojection_p95_px']:.4f} px.
- Held-out matches: {len(hp):,}. Square-root Sampson error median, prior → refined: {np.median(hp):.4f} → {np.median(hf):.4f} px; P90: {np.percentile(hp,90):.4f} → {np.percentile(hf,90):.4f} px.
- Held-out maximum error, prior → refined: {np.max(hp):.3f} → {np.max(hf):.3f} px. This maximum reports the tail alongside the median and P90.
- Translation change median / max: {np.median(shift):.4f} / {shift.max():.4f} m; rotation change median / max: {np.median(angle):.4f} / {angle.max():.4f} degrees.
- Reported local/global BA solves: {metrics['local_ba_solves']} / {metrics['global_ba_solves']}; final cleanup cycles: {metrics['cleanup_cycles']}. These are exported backend counters, not a complete timing or all-stage iteration trace.

The executable does not export pre-BA landmarks or initial solver cost. The CSV's prior-pose reprojection metric evaluates the **final landmarks with prior poses**; it is not the original pre-BA residual. Held-out epipolar errors compare the same reserved matches before/after, but do not establish absolute accuracy or ground-truth pose error. Pose changes measure correction, not accuracy. The pipeline's `pass` flag indicates successful output validation.

![BA performance](ba_performance.png)

## Per-frame strong landmark distribution

A strong observation has final reprojection error ≤1 px and belongs to a landmark seen in ≥4 exported images. Coverage counts occupied cells of an 8×6 image grid. A diagnostic weak flag means fewer than 50 strong landmarks or fewer than 16 occupied cells. These thresholds are reporting heuristics, not a training guarantee. Flagged frames: {len(summary['weak_frames'])}/{n}.

![Per-frame support and BA quality](per_frame_quality.png)

![Spatial distribution of strong landmarks](spatial_distribution.png)

[All frame metrics (CSV)](per_frame.csv) · [Machine-readable summary](summary.json)

## Twenty frames with weakest spatial support

| Image ID | All landmarks | Strong | Occupied cells / 48 | Final RMSE px |
|---|---:|---:|---:|---:|
'''
    for r in weakest: text+=f"| [{r['image_id']}](frames/{r['image_id']:010d}.jpg) | {r['landmarks']} | {r['strong_landmarks']} | {r['strong_occupied_cells']} | {r['final_rmse_px']:.3f} |\n"
    text+='\n## Every selected frame\n\nGreen points are strong observations; orange points are other retained landmarks.\n\n'
    for r in rows: text+=f"- [Image {r['image_id']}](frames/{r['image_id']:010d}.jpg): {r['strong_landmarks']} strong landmarks, {r['strong_occupied_cells']}/48 cells.\n"
    (report/'REPORT.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k!='pipeline'},indent=2))

if __name__=='__main__': main()
