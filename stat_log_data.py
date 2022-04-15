from glob import glob 
import numpy as np 
import json 
import pickle as pkl

LOG_DIRS = ['/home/mandi/mosaic/log_data', '/home/mandi/mosaic/baseline_data']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', '-k', default='*')
    args = parser.parse_args()

    runs = []
    for ldir in LOG_DIRS: 
        runs.extend( glob(ldir + f'/{args.keyword}') )

    for r in runs:
        print(f"Run: {r.split('/')[-1]}")
        result_files = glob(r + '/results*/*/test_across_*.json')
        toprint = ''
        for f in result_files:
            f = open(f)
            dic = json.load(f)
            toprint += " [step {}]: {:3f}, N={} ".format(dic['model_saved'], dic['success'], dic['N'])
        print(toprint, '\n')
