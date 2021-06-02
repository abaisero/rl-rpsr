#!/usr/bin/env python
import argparse

import pandas as pd


def main_eval(args):
    results = pd.read_csv(args.results, sep=' ', index_col=False)

    results_baseline = results.iloc[:, 0]
    errors = pd.DataFrame(
        {name: results_baseline - results[name] for name in results.columns}
    )
    MSE = errors.pow(2).mean().pow(0.5)

    # print(' '.join(results.columns))
    for name in results.columns:
        print(f'{MSE[name]: .3f}', end=' ')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')
    parser.add_argument('results')

    main_eval(parser.parse_args())


if __name__ == '__main__':
    main()
