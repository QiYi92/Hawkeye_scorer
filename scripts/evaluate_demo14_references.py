#!/usr/bin/env python3
"""Sparse visual-reference check; never turns these into independent ground truth."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def observed_distance(point,ring):
    """Distance to visible segments only; do not bridge missing or predicted arcs."""
    distances=[];entries=ring['samples'];point=np.array(point)
    for i,sample in enumerate(entries):
        following=entries[(i+1)%len(entries)]
        if sample['point'] is None or sample['source']!='observed':continue
        a=np.array(sample['point']);distances.append(float(np.linalg.norm(point-a)))
        if following['point'] is None or following['source']!='observed':continue
        b=np.array(following['point']);d=b-a
        if np.linalg.norm(d)>32:continue
        t=np.clip((point-a)@d/max(d@d,1e-9),0,1)
        distances.append(float(np.linalg.norm(point-(a+t*d))))
    return min(distances,default=float('inf'))


def evaluate(directory,references):
    rows=[]
    for ref in references['frames']:
        path=directory/ref['video']/f'frame_{ref["frame_index"]:04d}.json'
        if not path.exists():continue
        r=json.loads(path.read_text());center=r['center']
        error=float(np.linalg.norm(np.array(center['point'])-ref['center'])) if center['source']=='observed' else None
        distances=[]
        for mark in ref['ring_points']:
            candidates=[ring for ring in r['rings'] if mark['ring_id'] is None or ring['ring_id']==mark['ring_id']]
            d=min((observed_distance(mark['point'],ring) for ring in candidates),default=float('inf'))
            distances.append(d)
        # Report ALL finite distances, including misses, plus thresholded coverage.
        finite=[v for v in distances if np.isfinite(v)]
        rows.append(dict(video=ref['video'],frame_index=ref['frame_index'],center_error_px=error,
                         center_within_5px=error is not None and error<=5,
                         ring_reference_points=len(distances),
                         ring_median_distance_px=float(np.median(finite)) if finite else None,
                         ring_reference_coverage_5px=sum(v<=5 for v in distances)/len(distances) if distances else None,
                         ring_distances_px=[v if np.isfinite(v) else None for v in distances]))
    return dict(annotation_method=references['annotation_method'],
                reference_uncertainty_px=references['estimated_reference_uncertainty_px'],videos=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',type=Path,default=ROOT/'output/demo14/validation')
    p.add_argument('--references',type=Path,default=ROOT/'tests/fixtures/demo14_visual_references.json')
    args=p.parse_args();report=evaluate(args.validation,json.loads(args.references.read_text()))
    output=args.validation/'reference_metrics.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
