"""Batch evaluation uses exactly the same inference as predict.py and app.py."""
from dataclasses import asdict, replace
from pathlib import Path
from datetime import datetime, timezone
import platform
import sys
import json
import cv2
import numpy as np
from .io import ROOT, resolve, read_csv, write_csv, write_json, load_scene, load_params, read_image, new_run, digest
from .pipeline import count_image
from .metrics import counting_metrics
from .evaluation_guard import authorize, audit
from .visualization import save_prediction

REVIEWED = {'human_labeled','reviewed','peer_reviewed'}


def evaluate(config, split='dev', manifest=None, compare_scratch=False, include_draft=False, root=ROOT):
    tracker={}
    try:
        return _evaluate(config,split,manifest,compare_scratch,include_draft,root,tracker)
    except Exception as error:
        if tracker.get('request'):
            audit(dict(request_id=tracker['request'],status='failed',reason=str(error)),root)
        raise


def _evaluate(config,split,manifest,compare_scratch,include_draft,root,tracker):
    config=resolve(config,root)
    # Refuse explicit test/all requests even before reading any test image/labels.
    request=authorize([],config,split,root)
    tracker['request']=request
    if manifest:
        rows=read_csv(resolve(manifest,root))
    elif split=='all':
        rows=read_csv(root/'data/week1/manifests/dev.csv')+read_csv(root/'data/week1/manifests/test.csv')
    else:
        rows=read_csv(root/'data/week1/manifests'/f'{split}.csv')
    if not rows or any(not {'image_id','image','scene_id','split','sha256'} <= r.keys() for r in rows):
        raise ValueError('Manifest rỗng hoặc sai schema')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest có image_id trùng')
    if request is None:
        request=authorize(rows,config,split,root)
        tracker['request']=request
    if include_draft and request:
        raise ValueError('Không cho dùng nhãn draft trên test/holdout')
    if split!='all' and any(r['split']!=split for r in rows):
        raise ValueError('Nội dung manifest không khớp split yêu cầu')
    params,meta=load_params(config,root)
    labels={r['image_id']:r for r in read_csv(root/'data/week1/annotations/counts.csv')}
    path=new_run('experiments',root)
    (path/'config.toml').write_bytes(config.read_bytes())
    if request:
        (path/'freeze_snapshot.json').write_bytes((root/'configs/test_freeze.json').read_bytes())
    predictions=[]; errors=[]; scenes={}; mask_ious=[]
    try:
        for index,row in enumerate(rows):
            try:
                image_path=resolve(row['image'],root)
                if digest(image_path)!=row['sha256']:
                    raise ValueError('Ảnh đã đổi so với manifest')
                if row['scene_id'] not in scenes:
                    scenes[row['scene_id']]=load_scene(row['scene_id'],root)
                scene,bg,roi=scenes[row['scene_id']]
                image=read_image(image_path)
                result=count_image(image,bg,params,roi)
                label=labels.get(row['image_id'],{})
                accepted=label.get('review_status') in REVIEWED
                gt=int(label['count']) if label.get('count','')!='' and (accepted or include_draft) else None
                output=dict(image_id=row['image_id'],scene_id=row['scene_id'],split=row['split'],
                    condition=row.get('condition','original'),gt=gt,pred_cv=result.count,
                    pred_scratch='',label_status=label.get('review_status','missing'),
                    pipeline_ms=result.timings_ms['pipeline'],error='' if gt is None else result.count-gt)
                if compare_scratch:
                    scratch=count_image(image,bg,replace(params,backend='scratch',split_merged=False),roi)
                    output['pred_scratch']=scratch.count
                    a=result.stages['mask']>0; b=scratch.stages['mask']>0
                    mask_ious.append(float((a&b).sum()/max((a|b).sum(),1)))
                predictions.append(output)
                # Only dev illustrations before freeze; callers have passed the guard.
                if index<6:
                    save_prediction(path/'examples'/row['image_id'],image,result,dict(scene_id=row['scene_id']))
            except (ValueError,OSError,cv2.error,KeyError) as error:
                errors.append(dict(image_id=row['image_id'],error=str(error)))
        fields=['image_id','scene_id','split','condition','gt','pred_cv','pred_scratch','label_status','pipeline_ms','error']
        write_csv(path/'predictions.csv',predictions,fields)
        if errors:
            write_csv(path/'errors.csv',errors,['image_id','error'])
        labelled=[r for r in predictions if r['gt'] is not None]
        summary=dict(requested_images=len(rows),processed_images=len(predictions),failed_images=len(errors),
                     labelled_images=len(labelled),unreviewed_or_missing=len(predictions)-len(labelled),
                     label_policy='DRAFT INCLUDED: provisional' if include_draft else 'human-labelled/reviewed only',
                     background_status={sid:x[0].get('background_status') for sid,x in scenes.items()},
                     config=meta.get('experiment',{}),params=asdict(params))
        summary['overall']=counting_metrics([r['pred_cv'] for r in labelled],[r['gt'] for r in labelled]) if labelled else None
        summary['by_scene']={}
        for scene in sorted({r['scene_id'] for r in predictions}):
            subset=[r for r in labelled if r['scene_id']==scene]
            summary['by_scene'][scene]=counting_metrics([r['pred_cv'] for r in subset],[r['gt'] for r in subset]) if subset else None
        if compare_scratch and labelled:
            summary['scratch']=counting_metrics([r['pred_scratch'] for r in labelled],[r['gt'] for r in labelled])
            summary['implementation_mask_agreement']=dict(N=len(mask_ious),IoU=float(np.mean(mask_ious)) if mask_ious else None,
                                                         scope='all processed images, not ground-truth accuracy')
        write_json(path/'metrics.json',summary)
        write_json(path/'environment.json',dict(python=sys.version,platform=platform.platform(),opencv=cv2.__version__,
            numpy=np.__version__,measured_scope='count_image: preprocessing+segmentation+regions; excludes I/O, statistics and rendering'))
        if request:
            audit(dict(request_id=request,status='completed' if not errors else 'completed_with_errors',run_id=path.name),root)
        return path,summary
    except Exception as error:
        if request:
            audit(dict(request_id=request,status='failed',run_id=path.name,reason=str(error)),root)
            tracker['request']=None
        raise
