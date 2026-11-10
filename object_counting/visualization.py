"""Presentation only; no counting logic or label reads."""
from pathlib import Path
import cv2
import numpy as np
from .io import write_json


def overlay(image, prediction):
    out = image.copy()
    if prediction.contours:
        cv2.drawContours(out, prediction.contours, -1, (50,210,80), 1)
    for i, region in enumerate(prediction.regions, 1):
        x,y,w,h = region['bbox']
        cv2.rectangle(out, (x,y), (x+w,y+h), (40,160,255), 1)
        cv2.putText(out, str(i), (x,max(12,y-4)), cv2.FONT_HERSHEY_SIMPLEX,.45,(40,160,255),1)
    cv2.rectangle(out, (0,0), (min(out.shape[1],230),32), (30,40,42), -1)
    cv2.putText(out, f'Count: {prediction.count}', (9,23), cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    return out


def panel(image, prediction, width=280):
    tiles = []
    for title, value in [('Original',image), ('Gray',prediction.stages['gray']),
                          ('Difference',prediction.stages['diff']), ('Raw mask',prediction.stages['mask_raw']),
                          ('Clean mask',prediction.stages['mask']), ('Result',overlay(image,prediction))]:
        color = cv2.cvtColor(value,cv2.COLOR_GRAY2BGR) if value.ndim==2 else value.copy()
        color = cv2.resize(color,(width,int(width*color.shape[0]/color.shape[1])))
        cv2.rectangle(color,(0,0),(width,24),(30,40,42),-1)
        cv2.putText(color,title,(8,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        tiles.append(color)
    return np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])


def save_prediction(path, image, prediction, metadata=None):
    path=Path(path); (path/'stages').mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path/'overlay.png'),overlay(image,prediction))
    cv2.imwrite(str(path/'panel.png'),panel(image,prediction))
    for name, value in prediction.stages.items():
        cv2.imwrite(str(path/'stages'/f'{name}.png'),value)
    result = dict(count=prediction.count, regions=prediction.regions, params=prediction.params,
                  timings_ms=prediction.timings_ms, statistics=prediction.statistics,
                  processing_size=[image.shape[1],image.shape[0]], metadata=metadata or {})
    write_json(path/'result.json', result)
    return path
