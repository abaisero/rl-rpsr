#!/usr/bin/env python
import argparse
import logging

import numpy as np
import pandas as pd
from rl_psr import psr, rpsr
from rl_psr.pomdp import POMDP_Model
from rl_psr.serializer import IntentsSerializer, TestsSerializer


def rmse(data: np.ndarray):
    return np.sqrt((data ** 2).mean())


def main_info(args):
    pomdp_model = POMDP_Model.make(args.pomdp)

    if args.model == 'psr':
        Q = TestsSerializer().load(args.load_core)
        model = psr.PSR_Model(pomdp_model, Q)

    elif args.model == 'rpsr':
        I = IntentsSerializer().load(args.load_core)
        model = rpsr.RPSR_Model(pomdp_model, I)

    model_R = model.R_as_pomdp()

    if args.outcome:
        print(model.V)

    if args.stats:
        error = model_R - pomdp_model.R

        print(f'rank: {model.rank}')
        print(
            f'range: {error.min().round(args.decimals)} {error.max().round(args.decimals)}'
        )
        linfe = np.absolute(error).max()
        linfe_rel = linfe / np.absolute(pomdp_model.R).max()
        print(f'l-inf-e: {linfe.round(args.decimals)}')
        print(f'l-inf-e-rel: {linfe_rel.round(args.decimals)}')
        print(f'rmse: {rmse(error).round(args.decimals)}')

    if args.comparison:
        data = np.hstack([pomdp_model.R, model_R])
        index = pomdp_model.states
        columns = pd.MultiIndex.from_product(
            [['pomdp', args.model], pomdp_model.actions],
            names=['model', 'action'],
        )
        df = pd.DataFrame(data, index=index, columns=columns)
        print(df.round(args.decimals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')
    parser.add_argument('model', choices=['psr', 'rpsr'])
    parser.add_argument('--load-core', required=True)
    parser.add_argument('--outcome', action='store_true')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--comparison', action='store_true')
    parser.add_argument('--decimals', type=int, default=100)

    parser.add_argument('--log-filename', default=None)
    parser.add_argument(
        '--log-level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        default='INFO',
    )

    args = parser.parse_args()

    if args.log_filename is not None:
        logging.basicConfig(
            filename=args.log_filename,
            datefmt='%Y/%m/%d %H:%M:%S',
            format='%(asctime)s %(relativeCreated)d %(levelname)-8s %(name)-12s %(funcName)s - %(message)s',
            level=getattr(logging, args.log_level),
        )

    main_info(args)


if __name__ == '__main__':
    main()
