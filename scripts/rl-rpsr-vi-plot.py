#!/usr/bin/env python
import argparse
import glob
import logging

import matplotlib.pyplot as plt
from rl_rpsr import bsr, psr, rpsr
from rl_rpsr.metrics import VF_Metric
from rl_rpsr.pomdp import POMDP_Model
from rl_rpsr.serializer import IntentsSerializer, TestsSerializer, VF_Serializer


def main_vi(args):
    logger = logging.getLogger(__name__)
    logger.info('rl-psr-vi with args %s', args)

    pomdp_model = POMDP_Model.make(args.pomdp)

    if args.model == 'bsr':
        model = bsr.BSR_Model(pomdp_model)

    elif args.model == 'psr':
        Q = TestsSerializer().load(args.load_core)
        model = psr.PSR_Model(pomdp_model, Q)

    elif args.model == 'rpsr':
        I = IntentsSerializer().load(args.load_core)
        model = rpsr.RPSR_Model(pomdp_model, I)

    vf_serializer = VF_Serializer()
    metric = VF_Metric.factory(args.metric, start=model.start)

    pattern = f'{args.load_vf}.[0-9][0-9]*'
    filenames = glob.glob(pattern)
    horizons = (int(filename.split('.')[-1]) for filename in filenames)

    # zip, sort, unzip
    horizons, filenames = zip(*sorted(zip(horizons, filenames)))

    vf = vf_serializer.load(filenames[0])
    values, distances = [vf.value(model.start)], [float('nan')]
    for filename in filenames[1:]:
        vf_prev, vf = vf, vf_serializer.load(filename)

        values.append(vf.value(model.start))

        distance = metric.distance(vf_prev, vf)
        distances.append(distance)

    _, ax = plt.subplots()
    plt.title(f'VI Error - {args.pomdp.split("/")[-1]} - {args.model}')

    color, marker = 'blue', 'o'
    (line_distance,) = ax.plot(
        horizons, distances, linestyle='None', color=color, marker=marker,
    )
    ax.set_xlabel('horizon')
    ax.set_ylabel(f'distance ({args.metric})')
    ax.tick_params(axis='y', labelcolor=color)

    ax = ax.twinx()
    color, linestyle = 'red', '-'
    (line_value,) = ax.plot(horizons, values, color=color, linestyle=linestyle)
    ax.set_ylabel('value at start')
    ax.tick_params(axis='y', labelcolor=color)

    plt.legend((line_distance, line_value), ('distance', 'value'))

    if args.save_plot is not None:
        plt.savefig(args.save_plot, bbox_inches='tight')

    if args.show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')
    parser.add_argument('model', choices=['bsr', 'psr', 'rpsr'])
    parser.add_argument('--load-core', default=None)
    parser.add_argument('--load-vf', required=True)
    parser.add_argument(
        '--metric', choices=['alpha', 'bellman-at-start'], required=True
    )
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save-plot', default=None)

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
