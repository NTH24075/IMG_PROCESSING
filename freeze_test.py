"""Week 2 only: register reviewed baseline/final snapshots before first test run."""
import argparse
from object_counting.evaluation_guard import create_freeze


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',nargs='+',required=True)
    p.add_argument('--reviewers',nargs=2,required=True)
    a=p.parse_args()
    try:
        print(create_freeze(a.configs,a.reviewers))
    except (ValueError,OSError) as error:
        p.exit(2,str(error)+'\n')
