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
        label = str(i) if region.get('n', 1) == 1 else f"{i}:x{region['n']}"
        cv2.putText(out, label, (x,max(12,y-4)), cv2.FONT_HERSHEY_SIMPLEX,.45,(40,160,255),1)
    for cx, cy, r in getattr(prediction, 'statistics', {}).get('heads', []):
        cv2.circle(out, (cx, cy), r, (255,80,200), 1)
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
        if value.dtype != np.uint8:  # ratio / gradient maps are float32
            value = cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(path/'stages'/f'{name}.png'),value)
    result = dict(count=prediction.count, regions=prediction.regions, params=prediction.params,
                  timings_ms=prediction.timings_ms, statistics=prediction.statistics,
                  processing_size=[image.shape[1],image.shape[0]], metadata=metadata or {})
    write_json(path/'result.json', result)
    return path


def plot_mae_curves(table, folder):
    """One line per variant: MAE against degradation level, one figure per degradation kind."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    base={r['variant']:r['MAE'] for r in table if r['kind']=='none'}
    for kind in sorted({r['kind'] for r in table} - {'none'}):
        fig, ax = plt.subplots(figsize=(6,4))
        for variant in dict.fromkeys(r['variant'] for r in table):
            pts = sorted((float(r['level']), r['MAE']) for r in table if r['variant']==variant and r['kind']==kind)
            if pts:
                ax.plot(*zip(*pts), marker='o', label=variant)
                if variant in base:
                    ax.axhline(base[variant], ls=':', lw=.8, color=ax.lines[-1].get_color())
        ax.set(xlabel=f'{kind} level', ylabel='MAE (dev)', title=f'MAE vs {kind} (dotted: undegraded)')
        ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(folder/f'mae_{kind}.png', dpi=130); plt.close(fig)
    variants=list(dict.fromkeys(r['variant'] for r in table))
    if all(k in base for k in variants):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(variants, [base[v] for v in variants], color='#4a7bb7'); ax.set(ylabel='MAE (dev)', title='MAE per variant')
        ax.tick_params(axis='x', rotation=30); fig.tight_layout(); fig.savefig(folder/'mae_variants.png', dpi=130); plt.close(fig)
