#!/usr/bin/env python3
"""Export COLMAP camera centers as a viewer robot path (no floor projection)."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
source = root / 'data/kitti/images.txt'
poses = []
expect_pose = True
for line in source.read_text().splitlines():
    if line.lstrip().startswith('#'):
        continue
    if not expect_pose:
        expect_pose = True
        continue
    if not line.strip():
        continue
    fields = line.split()
    if len(fields) != 10:
        raise ValueError('Expected COLMAP image pose: ' + line)
    q = np.array([float(x) for x in fields[1:5]])
    t = np.array([float(x) for x in fields[5:8]])
    if not np.isfinite(q).all() or not np.isfinite(t).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid pose')
    w,x,y,z = q / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    center = -R.T @ t
    assert np.linalg.norm(R @ center + t) < 1e-8
    poses.append((fields[9], int(fields[0]), center))
    expect_pose = False
poses.sort(key=lambda p: p[0])  # KITTI zero-padded frame names are temporal order.
if len({p[0] for p in poses}) != len(poses) or len(poses) < 2:
    raise ValueError('Duplicate frames or insufficient poses')
points = [{'index': i, 'x': float(p[2][0]), 'y': float(p[2][1]),
           'z': float(p[2][2])} for i,p in enumerate(poses)]
output = source.with_name('camera_robot_path.json')
output.write_text(json.dumps({'version': 1, 'coordinate_frame': '3dgs_world',
    'path_type': 'ordered_polyline', 'closed': False, 'height_mode': 'recorded',
    'suggested_interpolation': 'linear_arc_length', 'point_count': len(points),
    'source': 'images.txt: camera centers -R_cw^T t_cw; ordered by image name',
    'points': points}, indent=2) + '\n')
a = np.array([p[2] for p in poses])
print('Saved', output, 'with', len(points), 'points')
print('XYZ extent:', np.ptp(a, axis=0))
print('Length:', np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
for i in (0,30,60,90,119):
    if i < len(a): print(i, a[i])
