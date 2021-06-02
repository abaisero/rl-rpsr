#!/usr/bin/env python
import argparse
import logging

from rl_psr import bsr, psr, rpsr
from rl_psr.metrics import VF_Metric
from rl_psr.pomdp import POMDP_Model
from rl_psr.serializer import (
    AlphaSerializer,
    IntentsSerializer,
    TestsSerializer,
    VF_Serializer,
)
from rl_psr.util import VI_Type


def main_vi(args):
    logger = logging.getLogger(__name__)
    logger.info('rl-psr-vi with args %s', args)

    pomdp_model = POMDP_Model.make(args.pomdp)

    if args.model == 'bsr':
        model = bsr.BSR_Model(pomdp_model)
        vi_algo = bsr.vi_factory(args.vi_type)

    elif args.model == 'psr':
        Q = TestsSerializer().load(args.load_core)
        model = psr.PSR_Model(pomdp_model, Q)
        vi_algo = psr.vi_factory(args.vi_type)

    elif args.model == 'rpsr':
        I = IntentsSerializer().load(args.load_core)
        model = rpsr.RPSR_Model(pomdp_model, I)
        vi_algo = rpsr.vi_factory(args.vi_type)

    vf = None
    if args.load_vf:
        logger.info(f'attempting to load value function from `{args.load_vf}`')
        try:
            vf = VF_Serializer().load(args.load_vf)
        except FileNotFoundError:
            logger.info('file not found')

    if vf is None:
        logger.info('initializing vf from vi_algo.init()')
        vf = vi_algo.init(model)

    vf_serializer = VF_Serializer()
    alpha_serializer = AlphaSerializer()

    metric = VF_Metric.factory(args.metric, start=model.start)

    eps = 1e-15

    logger.info('VI START')
    for _ in range(args.horizon):
        vf_prev, vf = vf, vi_algo.iterate(model, vf, eps=eps)
        logger.info(
            'VI iter horizon %d -> %d num_alphas %d -> %d',
            vf_prev.horizon,
            vf.horizon,
            len(vf_prev),
            len(vf),
        )

        distance = metric.distance(vf_prev, vf)
        logger.info('VI iter distance %f', distance)

        if args.save_vf is not None:
            filename = f'{args.load_vf}.{vf.horizon}'
            logger.info('saving vf to %s', filename)
            vf_serializer.dump(filename, vf)

        if args.save_alpha is not None:
            filename = f'{args.save_alpha}.{vf.horizon}'
            logger.info('saving alphas to %s', filename)
            alpha_serializer.dump(filename, vf)

    logger.info('VI STOP')

    if args.save_vf is not None:
        logger.info('saving vf to %s', args.save_vf)
        vf_serializer.dump(args.save_vf, vf)

    if args.save_alpha is not None:
        logger.info('saving alphas to %s', args.save_alpha)
        alpha_serializer.dump(args.save_alpha, vf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')
    parser.add_argument('model', choices=['bsr', 'psr', 'rpsr'])
    parser.add_argument('--load-core', default=None)
    parser.add_argument('--load-vf', default=None)
    parser.add_argument('--save-vf', default=None)
    parser.add_argument('--save-alpha', default=None)
    parser.add_argument('--disable-pbar', action='store_true')
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument(
        '--metric', choices=['alpha', 'bellman-at-start'], default=None
    )
    parser.add_argument(
        '--vi-type',
        type=VI_Type.__getitem__,
        choices=VI_Type.__members__.values(),
        default='TRUE_INC_PRUNING',
    )

    parser.add_argument('--log-filename', default=None)
    parser.add_argument(
        '--log-level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        default='INFO',
    )

    args = parser.parse_args()

    if not (args.model == 'bsr') == (args.load_core is None):
        parser.error(
            'The --load-core option is required iff the model is `bsr`'
        )

    if args.log_filename is not None:
        logging.basicConfig(
            filename=args.log_filename,
            datefmt='%Y/%m/%d %H:%M:%S',
            format='%(asctime)s %(relativeCreated)d %(levelname)-8s %(name)-12s %(funcName)s - %(message)s',
            level=getattr(logging, args.log_level),
        )

    main_vi(args)


if __name__ == '__main__':
    main()
