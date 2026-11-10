"""Check input integrity and label review coverage without running predictions."""
from collections import Counter
from object_counting.io import ROOT, read_csv, digest, resolve, load_scene, write_json


def validate():
    groups={split:read_csv(ROOT/'data/week1/manifests'/f'{split}.csv') for split in ('dev','test','legacy_holdout')}
    labels=read_csv(ROOT/'data/week1/annotations/counts.csv')
    ids=[r['image_id'] for rows in groups.values() for r in rows]
    if len(ids)!=len(set(ids)) or len(labels)!=len({r['image_id'] for r in labels}):
        raise ValueError('Duplicate image IDs')
    if set(ids)!={r['image_id'] for r in labels}:
        raise ValueError('Manifest/label IDs do not match')
    scene_splits={}; hashes={}
    for split,rows in groups.items():
        for row in rows:
            if row['split']!=split or digest(resolve(row['image']))!=row['sha256']:
                raise ValueError('Split/hash mismatch: '+row['image_id'])
            if row['sha256'] in hashes and hashes[row['sha256']]!=split:
                raise ValueError('Exact duplicate across splits: '+row['image_id'])
            hashes[row['sha256']]=split
            if split!='legacy_holdout':
                old=scene_splits.setdefault(row['scene_id'],split)
                if old!=split:raise ValueError('Scene crosses dev/test')
    backgrounds={scene:load_scene(scene)[0]['background_status'] for scene in scene_splits}
    active_ids={r['image_id'] for split in ('dev','test') for r in groups[split]}
    audit_path=ROOT/'outputs/test_audit.jsonl'
    import json
    events=[json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()] if audit_path.exists() else []
    test_started=any(e.get('status')=='allowed' and e.get('purpose')=='evaluation' for e in events)
    summary=dict(images={k:len(v) for k,v in groups.items()},review_status=dict(Counter(r['review_status'] for r in labels)),
                 backgrounds=backgrounds,scene_split=scene_splits,
                 technical_integrity='pass',human_review_complete=all(r['review_status']=='peer_reviewed' for r in labels if r['image_id'] in active_ids),
                 test_evaluation_run=test_started)
    write_json(ROOT/'data/week1/validation.json',summary)
    return summary


if __name__=='__main__':
    import json
    print(json.dumps(validate(),ensure_ascii=False,indent=2))
