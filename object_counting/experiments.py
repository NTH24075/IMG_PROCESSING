"""Batch evaluation uses exactly the same inference as predict.py and app.py.

Light freeze (v3 §6): any manifest touching a test image_id/sha256 runs only from a clean tree whose HEAD is the
latest freeze-test[-N] tag and whose config is committed. Each test run is appended to outputs/test_runs.log
before processing. This is a guard against accidents, not a security boundary.
"""
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import platform
import re
import subprocess
import sys
import tomllib
import cv2
import numpy as np
from .io import ROOT, resolve, read_csv, write_csv, write_json, load_scene, load_params, read_image, new_run, digest, load_points
from .pipeline import count_image
from .metrics import counting_metrics, error_types
from .degrade import degrade
from .visualization import save_prediction

REVIEWED = {'human_labeled','reviewed','peer_reviewed'}
TEST_SPLITS = ('test','legacy_holdout')
ERROR_KINDS = ('merge','split','false_blob','miss')


def _git(root, *args):
    try:
        return subprocess.run(['git',*args],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError) as error:
        raise ValueError('git '+' '.join(args)+' thất bại: '+str(getattr(error,'stderr','') or error).strip()) from error


def touches_test(rows, root=ROOT):
    ids, hashes = set(), set()
    for split in TEST_SPLITS:
        path = root/'data/week1/manifests'/f'{split}.csv'
        for r in read_csv(path) if path.exists() else []:
            ids.add(r['image_id']); hashes.add(r['sha256'])
    return any(r.get('split') in TEST_SPLITS or r['image_id'] in ids or r.get('source_image_id') in ids or r.get('sha256') in hashes
               for r in rows)


def check_freeze(config, root=ROOT):
    if _git(root,'status','--porcelain'):
        raise ValueError('TEST ĐANG KHÓA: working tree chưa sạch (commit hoặc bỏ thay đổi trước).')
    tags = [t for t in _git(root,'tag','--list','freeze-test*').split() if re.fullmatch(r'freeze-test(-\d+)?',t)]
    if not tags:
        raise ValueError('TEST ĐANG KHÓA: chưa có tag freeze-test.')
    latest = max(tags,key=lambda t:int(t[12:]) if len(t)>11 else 1)
    head = _git(root,'rev-parse','HEAD')
    if _git(root,'rev-list','-n','1',latest) != head:
        raise ValueError(f'TEST ĐANG KHÓA: HEAD không trùng tag mới nhất {latest}.')
    _git(root,'ls-files','--error-unmatch',str(resolve(config,root).relative_to(root)))
    return dict(commit=head,tag=latest)


def log_test_run(root, **fields):
    path = root/'outputs/test_runs.log'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),**fields),ensure_ascii=False)+'\n')


def git_state(root=ROOT):
    try:
        return dict(commit=_git(root,'rev-parse','HEAD'),tags=_git(root,'tag','--points-at','HEAD').split(),
                    dirty=bool(_git(root,'status','--porcelain')))
    except ValueError as error:
        return dict(error=str(error))


def resolved_toml(params, meta):
    head = ''.join(f'{k} = {json.dumps(v)}\n' for k,v in meta.get('experiment',{}).items() if isinstance(v,(str,int,float,bool)))
    return '[experiment]\n'+head+'\n[processing]\n'+''.join(f'{k} = {json.dumps(v)}\n' for k,v in asdict(params).items())


def evaluate(config, split='dev', manifest=None, compare_scratch=False, include_draft=False, root=ROOT,
             overrides=None, kinds=None, examples=6):
    config = resolve(config,root)
    rows = read_csv(resolve(manifest,root)) if manifest else (
        read_csv(root/'data/week1/manifests/dev.csv')+read_csv(root/'data/week1/manifests/test.csv') if split=='all'
        else read_csv(root/'data/week1/manifests'/f'{split}.csv'))
    if kinds:
        rows = [r for r in rows if r.get('kind','none') in kinds]
    if not rows or any(not {'image_id','image','scene_id','split','sha256'} <= r.keys() for r in rows):
        raise ValueError('Manifest rỗng hoặc sai schema')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest có image_id trùng')
    if not manifest and split not in ('all',) and any(r['split']!=split for r in rows):
        raise ValueError('Nội dung manifest không khớp split yêu cầu')
    params, meta = load_params(config,root)
    if overrides:
        params = replace(params,**overrides); params.validate()
    freeze = None
    if touches_test(rows,root):
        if include_draft:
            raise ValueError('Không cho dùng nhãn draft trên test/holdout')
        freeze = check_freeze(config,root)
        # Written BEFORE any test image is read; if this fails nothing runs.
        log_test_run(root,event='start',config=str(config.relative_to(root)),manifest=str(manifest or split),
                     overrides=overrides or {},**freeze)
    labels = {r['image_id']:r for r in read_csv(root/'data/week1/annotations/counts.csv')}
    points = load_points(root)
    path = new_run('experiments',root)
    (path/'config.toml').write_text(resolved_toml(params,meta),encoding='utf-8')
    write_json(path/'git.json',dict(git_state(root),freeze=freeze))
    predictions=[]; errors=[]; diagnosis=[]; scenes={}; mask_ious=[]
    for index,row in enumerate(rows):
        try:
            image_path = resolve(row['image'],root)
            if digest(image_path)!=row['sha256']:
                raise ValueError('Ảnh đã đổi so với manifest')
            if row['scene_id'] not in scenes:
                scenes[row['scene_id']] = load_scene(row['scene_id'],root)
            scene,bg,roi = scenes[row['scene_id']]
            image = read_image(image_path)
            kind, level = row.get('kind') or 'none', row.get('level') or ''
            if kind != 'none':
                image = degrade(image,kind,float(level),int(row['seed']))
            result = count_image(image,bg,params,roi,scene.get('perspective'))
            source = row.get('source_image_id') or row['image_id']
            label = labels.get(source,{})
            accepted = label.get('review_status') in REVIEWED
            gt = int(label['count']) if label.get('count','')!='' and (accepted or include_draft) else None
            output = dict(image_id=row['image_id'],source_image_id=source,scene_id=row['scene_id'],split=row['split'],
                condition=row.get('condition','original'),kind=kind,level=level,gt=gt,pred_cv=result.count,
                pred_scratch='',label_status=label.get('review_status','missing'),
                pipeline_ms=result.timings_ms['pipeline'],error='' if gt is None else result.count-gt)
            if source in points:
                inside = [(x,y) for x,y in points[source] if roi is None or roi[int(y),int(x)]]
                kinds_found = error_types(result.regions,inside)
                output.update(kinds_found); diagnosis.append(dict(image_id=row['image_id'],kind=kind,level=level,points=len(inside),
                                                                   pred=result.count,**kinds_found))
            if compare_scratch:
                scratch = count_image(image,bg,replace(params,backend='scratch',split_merged=False),roi,scene.get('perspective'))
                output['pred_scratch'] = scratch.count
                a = result.stages['mask']>0; b = scratch.stages['mask']>0
                mask_ious.append(float((a&b).sum()/max((a|b).sum(),1)))
            predictions.append(output)
            if index<examples:
                save_prediction(path/'examples'/row['image_id'],image,result,dict(scene_id=row['scene_id']))
        except (ValueError,OSError,cv2.error,KeyError) as error:
            errors.append(dict(image_id=row['image_id'],error=str(error)))
    fields = ['image_id','source_image_id','scene_id','split','condition','kind','level','gt','pred_cv','pred_scratch',
              'label_status','pipeline_ms','error',*ERROR_KINDS]
    write_csv(path/'predictions.csv',[{k:r.get(k,'') for k in fields} for r in predictions],fields)
    if diagnosis:
        write_csv(path/'error_types.csv',diagnosis,diagnosis[0].keys())
    if errors:
        write_csv(path/'errors.csv',errors,['image_id','error'])
    labelled = [r for r in predictions if r['gt'] is not None]
    summary = dict(requested_images=len(rows),processed_images=len(predictions),failed_images=len(errors),
                   labelled_images=len(labelled),unreviewed_or_missing=len(predictions)-len(labelled),
                   label_policy='DRAFT INCLUDED: provisional' if include_draft else 'human-labelled/reviewed only',
                   background_status={sid:x[0].get('background_status') for sid,x in scenes.items()},
                   perspective={sid:x[0].get('perspective') for sid,x in scenes.items()},
                   config=meta.get('experiment',{}),params=asdict(params))
    summary['overall'] = counting_metrics([r['pred_cv'] for r in labelled],[r['gt'] for r in labelled]) if labelled else None
    summary['by_scene'] = {}; summary['by_condition'] = {}
    groups = defaultdict(list)
    for r in labelled:
        groups[f"{r['kind']}:{r['level']}" if r['kind']!='none' else 'none'].append(r)
    for scene in sorted({r['scene_id'] for r in predictions}):
        subset = [r for r in labelled if r['scene_id']==scene]
        summary['by_scene'][scene] = counting_metrics([r['pred_cv'] for r in subset],[r['gt'] for r in subset]) if subset else None
    for key,subset in groups.items():
        summary['by_condition'][key] = counting_metrics([r['pred_cv'] for r in subset],[r['gt'] for r in subset])
    if diagnosis:
        summary['error_types'] = {k:int(sum(d[k] for d in diagnosis)) for k in ERROR_KINDS}
    if compare_scratch and labelled:
        summary['scratch'] = counting_metrics([r['pred_scratch'] for r in labelled],[r['gt'] for r in labelled])
        summary['implementation_mask_agreement'] = dict(N=len(mask_ious),IoU=float(np.mean(mask_ious)) if mask_ious else None,
                                                        scope='all processed images, not ground-truth accuracy')
    write_json(path/'metrics.json',summary)
    write_json(path/'environment.json',dict(python=sys.version,platform=platform.platform(),opencv=cv2.__version__,
        numpy=np.__version__,measured_scope='count_image: preprocessing+segmentation+regions+counting; excludes I/O, degradation, statistics and rendering'))
    if freeze:
        log_test_run(root,event='done',run_id=path.name,MAE=(summary['overall'] or {}).get('MAE'),failed=len(errors),**freeze)
    return path,summary


def run_experiment(spec, root=ROOT):
    """configs/experiments/eN.toml: [experiment] base/manifest/kinds + one [variants.<name>] table per variant."""
    spec = resolve(spec,root)
    with spec.open('rb') as f:
        plan = tomllib.load(f)
    head, variants = plan['experiment'], plan.get('variants',{})
    if not variants:
        raise ValueError('Experiment needs at least one [variants.<name>] table')
    destination = new_run('experiments',root)
    (destination/'experiment.toml').write_bytes(spec.read_bytes())
    table = []
    for name,overrides in variants.items():
        run,summary = evaluate(head['base'],head.get('split','dev'),head.get('manifest'),root=root,overrides=overrides,
                               kinds=head.get('kinds'),examples=0)
        for key,m in sorted(summary['by_condition'].items()):
            kind,_,level = key.partition(':')
            table.append(dict(variant=name,run_id=run.name,kind=kind,level=level,N=m['N'],MAE=m['MAE'],RMSE=m['RMSE'],
                              Bias=m['Bias'],Acc=m['Acc'],**{k:'' for k in ERROR_KINDS}))
        if summary.get('error_types') and len(summary['by_condition'])==1:
            table[-1].update(summary['error_types'])
    write_csv(destination/'summary.csv',table,table[0].keys())
    from .visualization import plot_mae_curves
    plot_mae_curves(table,destination/'figures')
    return destination,table
