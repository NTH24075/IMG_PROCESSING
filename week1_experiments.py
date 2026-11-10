"""Week-1 feasibility checks on DEV only; never chooses a final test configuration."""
from dataclasses import asdict
from time import perf_counter
import json
import cv2
import numpy as np
from object_counting.io import ROOT, load_params, load_scene, read_image, resolve, read_csv, write_csv, write_json, new_run
from object_counting.pipeline import count_image
from object_counting.experiments import evaluate
from object_counting.preprocessing import add_noise, histogram
from object_counting.segmentation import edge_illustrations
from object_counting.visualization import panel


def main():
    variants=[('baseline',{}),('gamma',{'enhancement':'gamma','gamma':.7}),
              ('percentile',{'enhancement':'percentile'}),('median',{'filter':'median'}),
              ('fixed_threshold',{'thresh':25}),('no_morphology',{'open_k':1,'close_k':1}),
              ('watershed',{'split_merged':True,'watershed_ratio':.5})]
    results=[]; base,_=load_params()
    for name,changes in variants:
        config=ROOT/'configs'/f'{name}.toml'
        if name!='baseline' and not config.exists():
            values=asdict(base);values.update(changes)
            text='[experiment]\nname = "'+name+'"\nrole = "variant"\nstatus = "development"\n\n[processing]\n'
            text+='\n'.join(k+' = '+json.dumps(v) for k,v in values.items())+'\n'
            config.write_text(text)
        output,summary=evaluate(config,'dev')
        row=dict(variant=name,run_id=output.name,processed=summary['processed_images'],
                 verified_label_count=summary['labelled_images'],scope='26 legacy human labels; new-scene XML drafts excluded')
        row.update({key:summary['overall'][key] for key in ('MAE','RMSE','Acc','Bias')})
        results.append(row)
    destination=new_run('benchmarks')
    write_csv(destination/'variants.csv',results,results[0].keys())
    dev=read_csv(ROOT/'data/week1/manifests/dev.csv')
    # Comparable timings on a small predeclared DEV sample, not on test.
    sample=[dev[4],dev[12],dev[30],dev[42]]; times=[]
    for row in sample:
        scene,bg,roi=load_scene(row['scene_id']); image=read_image(resolve(row['image']))
        for backend in ('opencv','scratch'):
            values=asdict(base);values['backend']=backend
            from object_counting.pipeline import Params
            p=Params(**values);count_image(image,bg,p,roi)  # warm-up excluded
            measures=[count_image(image,bg,p,roi).timings_ms['pipeline'] for _ in range(3)]
            times.append(dict(image_id=row['image_id'],backend=backend,median_pipeline_ms=float(np.median(measures)),repeats=3))
    write_csv(destination/'timings.csv',times,times[0].keys())
    row=dev[42];scene,bg,roi=load_scene(row['scene_id']); image=read_image(resolve(row['image']))
    for kind,amount in [('gaussian',15),('salt_pepper',.02)]:
        noisy=add_noise(image,kind,amount,42)
        cv2.imwrite(str(destination/f'noise_{kind}_seed42.png'),noisy)
        filtered=np.hstack([image,noisy,cv2.GaussianBlur(noisy,(3,3),0),cv2.medianBlur(noisy,3)])
        cv2.imwrite(str(destination/f'filter_comparison_{kind}.png'),filtered)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for name,edge in edge_illustrations(gray).items():
        cv2.imwrite(str(destination/f'{name}_illustration.png'),edge)
    h,cdf=histogram(gray)
    write_csv(destination/'histogram_cdf.csv',[dict(intensity=i,frequency=int(h[i]),cdf=float(cdf[i])) for i in range(256)],['intensity','frequency','cdf'])
    write_json(destination/'protocol.json',dict(split='dev',seed=42,noise_gaussian_sigma=15,noise_impulse_fraction=.02,
        caveat='Feasibility only. New labels and backgrounds need human review. No final config selected; no test predictions.',
        edge_parameters='Gaussian k=3; Sobel k=3; Canny 50/150; illustration only',
        timing_scope='pipeline stages excluding I/O/rendering/statistics; median of 3 runs after one warm-up'))
    print('Week-1 dev feasibility outputs:',destination)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
