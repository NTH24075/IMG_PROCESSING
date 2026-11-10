"""Operational test lock, not an adversarial security boundary."""
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid
from .io import ROOT, read_csv, digest, resolve, write_json, load_params


def audit(event, root=ROOT):
    path=root/'outputs/test_audit.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
    event=dict(timestamp=datetime.now(timezone.utc).isoformat(),**event)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush()


def protected_rows(root=ROOT):
    rows=[]
    for split in ('test','legacy_holdout'):
        path=root/'data/week1/manifests'/f'{split}.csv'
        if path.exists():
            rows.extend(read_csv(path))
    return rows


def is_protected(rows, split, root=ROOT):
    if split in ('test','all','legacy_holdout'):
        return True
    registry=protected_rows(root)
    ids={r['image_id'] for r in registry}; scenes={r['scene_id'] for r in registry if r.get('split')=='test'}
    paths={resolve(r['image'],root) for r in registry}; hashes={r.get('sha256') for r in registry}
    for row in rows:
        path=resolve(row['image'],root)
        if row.get('split') in ('test','legacy_holdout') or row.get('image_id') in ids or row.get('scene_id') in scenes or path in paths:
            return True
        if row.get('sha256') in hashes or (path.is_file() and digest(path) in hashes):
            return True
    return False


def snapshot(root=ROOT):
    files=set(root.glob('*.py')) | set((root/'object_counting').glob('*.py')) | set((root/'configs').rglob('*.toml'))
    files |= set((root/'data/week1').rglob('*.csv')) | set((root/'data/week1').rglob('*.json'))
    files |= {root/name for name in ('pyproject.toml','uv.lock') if (root/name).exists()}
    for manifest in (root/'data/week1/manifests').glob('*.csv'):
        for row in read_csv(manifest):
            path=resolve(row['image'],root)
            if row.get('sha256') and digest(path)!=row['sha256']:
                raise ValueError('Ảnh đã đổi so với manifest: '+row['image_id'])
            files.add(path)
    for scene_file in (root/'configs/scenes').glob('*.toml'):
        import tomllib
        scene=tomllib.loads(scene_file.read_text())['scene']
        files.add(resolve(scene['background'],root))
    return {str(path.relative_to(root)):digest(path) for path in sorted(files)}


def create_freeze(configs, reviewers, root=ROOT):
    if len({x.strip().casefold() for x in reviewers})<2 or any(not x.strip() for x in reviewers):
        raise ValueError('Freeze cần hai người duyệt khác nhau')
    labels=read_csv(root/'data/week1/annotations/counts.csv')
    active=set()
    for split in ('dev','test'):
        path=root/'data/week1/manifests'/f'{split}.csv'
        if path.exists():
            active.update(r['image_id'] for r in read_csv(path))
    label_by_id={r['image_id']:r for r in labels}
    if len(label_by_id)!=len(labels):
        raise ValueError('Nhãn có image_id trùng; chưa được freeze')
    if any(i not in label_by_id or label_by_id[i]['review_status']!='peer_reviewed' for i in active):
        raise ValueError('Chưa kiểm tra chéo toàn bộ nhãn dev/test; chưa được freeze')
    for image_id in active:
        label=label_by_id[image_id]
        if not label.get('reviewer','').strip() or not label.get('peer_reviewer','').strip() or label['reviewer'].casefold()==label['peer_reviewer'].casefold():
            raise ValueError('Thiếu hai người duyệt nhãn khác nhau: '+image_id)
        try:
            if int(label['count'])<0:
                raise ValueError()
        except (ValueError,KeyError):
            raise ValueError('Nhãn số đếm không hợp lệ: '+image_id)
    for scene_file in (root/'configs/scenes').glob('*.toml'):
        import tomllib
        scene=tomllib.loads(scene_file.read_text())['scene']
        if scene.get('background_status') not in ('reviewed','legacy_offline_median'):
            raise ValueError('Cần rà soát nền và cập nhật trạng thái reviewed: '+scene['id'])
        if scene.get('roi_proposal_status','reviewed')!='reviewed':
            raise ValueError('ROI vẫn là đề xuất chưa duyệt: '+scene['id'])
    path=root/'configs/test_freeze.json'
    if path.exists():
        raise ValueError('Bản freeze đã tồn tại; cần lưu lịch sử và lý do trước khi tạo bản mới')
    for config in configs:
        load_params(config,root)
    entries={str(resolve(c,root).relative_to(root)):digest(resolve(c,root)) for c in configs}
    if not entries:
        raise ValueError('Cần ít nhất một cấu hình')
    record=dict(status='finalized',allowed_splits=['test'],reviewers=reviewers,created_at=datetime.now(timezone.utc).isoformat(),
                configs=entries,snapshot=snapshot(root))
    write_json(path,record)
    return path


def authorize(rows, config, split, root=ROOT, purpose='evaluation'):
    if not is_protected(rows,split,root):
        return None
    request_id=uuid.uuid4().hex
    base=dict(request_id=request_id,purpose=purpose,requested_split=split,config=str(config))
    audit(dict(base,status='requested'),root)
    try:
        freeze_path=root/'configs/test_freeze.json'
        if not freeze_path.exists():
            raise ValueError('TEST ĐANG KHÓA: chưa có bản freeze. Chỉ dùng dev trong tuần 1.')
        record=json.loads(freeze_path.read_text())
        legacy=[r for r in protected_rows(root) if r.get('split')=='legacy_holdout']
        legacy_paths={resolve(r['image'],root) for r in legacy}; legacy_hashes={r.get('sha256') for r in legacy}
        uses_legacy=split=='legacy_holdout' or any(r.get('split')=='legacy_holdout' or resolve(r['image'],root) in legacy_paths or
            (resolve(r['image'],root).is_file() and digest(resolve(r['image'],root)) in legacy_hashes) for r in rows)
        if uses_legacy and 'legacy_holdout' not in record.get('allowed_splits',[]):
            raise ValueError('Holdout lịch sử cần protocol/freeze riêng, ngoài test chuyển cảnh chính')
        config_path=resolve(config,root); relative=str(config_path.relative_to(root))
        if record.get('status')!='finalized' or record.get('configs',{}).get(relative)!=digest(config_path):
            raise ValueError('Cấu hình chưa đăng ký hoặc đã đổi sau freeze')
        if record.get('snapshot')!=snapshot(root):
            raise ValueError('Mã, dữ liệu, nền hoặc cấu hình đã đổi sau freeze')
        audit(dict(base,status='allowed',freeze_sha256=digest(freeze_path),config_sha256=digest(config_path)),root)
    except (OSError,ValueError,KeyError) as error:
        audit(dict(base,status='denied',reason=str(error)),root)
        raise ValueError(str(error)) from error
    return request_id
