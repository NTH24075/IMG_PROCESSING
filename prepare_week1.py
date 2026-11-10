"""Prepare an isolated, versioned multi-scene dataset without overwriting vtest."""
from pathlib import Path
import csv
import json
import shutil
import uuid
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from object_counting.io import ROOT, digest, read_csv, write_csv, write_json

FIELDS = ['image_id','image','scene_id','split','source_id','source_frame','timestamp_seconds','condition','sha256']
LABEL_FIELDS = ['image_id','count','provenance','review_status','reviewer','peer_reviewer','notes']


def xml_objects(path):
    result = {}
    for frame in ET.parse(path).getroot().findall('frame'):
        objects=[]
        for obj in frame.findall('./objectlist/object'):
            box=obj.find('box')
            if box is not None:
                objects.append(dict(id=obj.get('id'), **{k:int(box.get(k)) for k in ('xc','yc','w','h')},
                                    appearance=obj.findtext('appearance','unknown')))
        result[int(frame.get('number'))]=objects
    return result


def prepare(root=ROOT):
    destination=root/'data/week1'
    if destination.exists():
        raise ValueError('data/week1 đã tồn tại. Không ghi đè nhãn đã rà soát; dùng dữ liệu hiện có.')
    if any((root/'configs/scenes'/f'scene_{i:02d}.toml').exists() for i in (1,2,3)):
        raise ValueError('Cấu hình cảnh đã tồn tại; không tự ghi đè')
    staging=root/'data'/('.week1_build_'+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    manifests={'dev':[], 'test':[], 'legacy_holdout':[]}; labels=[]; candidates={}; scene_configs=[]
    legacy=read_csv(root/'data/ground_truth.csv')
    for row in legacy:
        scene='scene_01'; image_id=f"{scene}_{Path(row['image']).stem}"
        file=root/'data/frames'/row['image']
        split='dev' if row['split']=='dev' else 'legacy_holdout'
        index=int(Path(row['image']).stem[1:])
        manifests[split].append(dict(image_id=image_id,image=file.relative_to(root).as_posix(),scene_id=scene,
            split=split,source_id='vtest',source_frame=index,timestamp_seconds=index/10,
            condition='original',sha256=digest(file)))
        labels.append(dict(image_id=image_id,count=row['count'],provenance='legacy_user_annotation',
                           review_status='human_labeled',reviewer='legacy_user',peer_reviewer='',notes='Giữ nguyên nhãn lịch sử; chưa khẳng định đã kiểm tra chéo.'))
    legacy_bg=cv2.imread(str(root/'data/background.png'))
    scene_configs.append(dict(id='scene_01',name='Vtest — ngoài trời',width=legacy_bg.shape[1],height=legacy_bg.shape[0],
        background='data/background.png',role='dev',background_status='legacy_offline_median',
        background_policy='Legacy median from full video; offline/transductive, not causal online initialization.'))
    # Prefix lengths and sample schedule are fixed before inspecting test outcomes.
    for scene,name,xml,split,prefix,start,step in [
        ('scene_02','Meet_Crowd','mc1gt.xml','dev',30,45,9),
        ('scene_03','WalkByShop1cor','cwbs1gt.xml','test',500,500,25),
    ]:
        annotations=xml_objects(root/'data/raw/caviar'/xml)
        frame_dir=staging/'frames'/scene; frame_dir.mkdir(parents=True)
        cap=cv2.VideoCapture(str(root/'data/raw/caviar'/f'{name}.mpg'))
        if not cap.isOpened():
            raise ValueError('Không đọc được '+name)
        fps=cap.get(cv2.CAP_PROP_FPS); frames_for_background=[]; i=0
        while True:
            ok,image=cap.read()
            if not ok: break
            if i<prefix and i % 5==0:
                frames_for_background.append(image.copy())
            if i>=start and (i-start)%step==0:
                image_id=f'{scene}_f{i:05d}'; output=frame_dir/f'f{i:05d}.png'
                cv2.imwrite(str(output),image)
                record=dict(image_id=image_id,image=f'data/week1/frames/{scene}/{output.name}',scene_id=scene,
                    split=split,source_id=name,source_frame=i,timestamp_seconds=round(i/fps,4),
                    condition='original',sha256=digest(output))
                manifests[split].append(record)
                objects=annotations.get(i,[]); candidates[image_id]=objects
                # XML is a proposal, not verified count ground truth.
                eligible=[obj for obj in objects if scene!='scene_03' or obj['yc']>=65]
                labels.append(dict(image_id=image_id,count=len(eligible),provenance=f'caviar_xml:{xml}:frame={i}',
                    review_status='draft_xml',reviewer='',peer_reviewer='',
                    notes='Cần rà soát ảnh, người đứng yên/che khuất, box ngoài ROI và căn chỉnh frame MPEG–XML.'))
            i+=1
        cap.release()
        if not frames_for_background or i<=start:
            raise ValueError('Video too short for fixed initialization protocol: '+name)
        bg=np.median(np.stack(frames_for_background),axis=0).astype(np.uint8)
        (staging/'backgrounds').mkdir(exist_ok=True)
        cv2.imwrite(str(staging/'backgrounds'/f'{scene}.png'),bg)
        scene_configs.append(dict(id=scene,name=name,width=bg.shape[1],height=bg.shape[0],role=split,
            background=f'data/week1/backgrounds/{scene}.png',background_status='needs_visual_review',
            background_policy=f'Median of every 5th decoded frame in prefix [0,{prefix}); samples start at {start}.',
            warmup_frames=prefix,decoded_frames=i,fps=fps))
    for split,rows in manifests.items():
        write_csv(staging/'manifests'/f'{split}.csv',rows,FIELDS)
    write_csv(staging/'annotations/counts.csv',labels,LABEL_FIELDS)
    write_json(staging/'annotations/xml_candidates.json',candidates)
    write_json(staging/'dataset.json',dict(version='week1',sources=scene_configs,
        counts={k:len(v) for k,v in manifests.items()},
        warning='New XML labels are draft; peer review/background review are not fabricated. No test prediction was run.'))
    staging.rename(destination)
    for scene in scene_configs:
        path=root/'configs/scenes'/f"{scene['id']}.toml"
        if path.exists():
            raise ValueError('Scene config exists; refusing overwrite: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        text='[scene]\n'+'\n'.join(f'{k} = {json.dumps(v,ensure_ascii=False)}' for k,v in scene.items())+'\n'
        if scene['id']=='scene_03':
            text+='roi = [[0, 65], [383, 65], [383, 287], [0, 287]]\nroi_proposal_status = "needs_group_review"\n'
        path.write_text(text)
    return destination, {k:len(v) for k,v in manifests.items()}


if __name__=='__main__':
    try:
        print(prepare())
    except (ValueError,OSError) as error:
        raise SystemExit(str(error))
