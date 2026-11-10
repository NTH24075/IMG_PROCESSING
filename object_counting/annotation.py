"""Explicit human-review records; XML import never auto-approves labels."""
from datetime import datetime, timezone
import json
import shutil
import uuid
from .io import ROOT, read_csv, write_csv


def review_label(image_id, count, reviewer, root=ROOT):
    reviewer=reviewer.strip()
    if not reviewer or type(count) is not int or count<0:
        raise ValueError('Cần tên người duyệt và số đếm nguyên không âm')
    path=root/'data/week1/annotations/counts.csv'
    rows=read_csv(path)
    matches=[r for r in rows if r['image_id']==image_id]
    if len(matches)!=1:
        raise ValueError('Không tìm được nhãn duy nhất cho '+image_id)
    row=matches[0]; before=dict(row)
    if row['count']==str(count) and row['review_status']=='peer_reviewed' and reviewer.casefold() in (row['reviewer'].casefold(),row['peer_reviewer'].casefold()):
        pass  # Unchanged confirmation does not erase an existing independent review.
    elif row.get('reviewer') and row['reviewer'].casefold()!=reviewer.casefold() and row['count']==str(count) and row['review_status'] in ('human_labeled','reviewed','peer_reviewed'):
        row['peer_reviewer']=reviewer; row['review_status']='peer_reviewed'
    else:
        # A correction needs a fresh second review; do not retain stale approval.
        row.update(reviewer=reviewer,peer_reviewer='',review_status='reviewed')
    row['count']=str(count)
    backup=path.parent/'backups'; backup.mkdir(exist_ok=True)
    shutil.copy2(path,backup/f'counts_{uuid.uuid4().hex}.csv')
    write_csv(path,rows,rows[0].keys())
    with (path.parent/'review_log.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),image_id=image_id,
                    reviewer=reviewer,before=before,after=row),ensure_ascii=False)+'\n')
    return row
