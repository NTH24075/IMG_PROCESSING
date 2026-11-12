import argparse
import json
from object_counting.experiments import evaluate, run_experiment


def main(argv=None):
    parser=argparse.ArgumentParser(description='Evaluate configured pipeline. Any manifest touching test images needs the freeze-test tag.')
    parser.add_argument('--config',default='configs/baseline.toml')
    parser.add_argument('--split',choices=['dev','test','all','legacy_holdout'],default='dev')
    parser.add_argument('--manifest',help='e.g. data/week1/manifests/robustness_dev.csv; overrides --split')
    parser.add_argument('--experiment',help='configs/experiments/eN.toml: run every variant and write summary.csv + figures')
    parser.add_argument('--compare-scratch',action='store_true')
    parser.add_argument('--scratch',type=int,choices=[0,1],help='Compatibility: 1 compares NumPy implementation')
    parser.add_argument('--include-draft',action='store_true',help='Dev only; explicitly provisional metrics using XML draft labels')
    args=parser.parse_args(argv)
    try:
        if args.experiment:
            path,table=run_experiment(args.experiment)
            for r in table:
                print(f"{r['variant']:<18}{r['kind']:<11}{r['level']:<6} N={r['N']:<4} MAE={r['MAE']:.3f} Bias={r['Bias']:+.3f}")
            print('Output:',path); return
        path,result=evaluate(args.config,args.split,args.manifest,args.compare_scratch or args.scratch==1,args.include_draft)
    except (ValueError,OSError,KeyError,TypeError) as error:
        parser.exit(2,str(error)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2)); print('Output:',path)
    if result['failed_images']:
        parser.exit(1,'Có ảnh xử lý lỗi; xem errors.csv, không coi đây là đánh giá đầy đủ.\n')


if __name__=='__main__':
    main()
