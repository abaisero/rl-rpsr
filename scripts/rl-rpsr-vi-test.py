#!/usr/bin/env python
import argparse
import logging

import numpy as np
from rl_rpsr import bsr, pomdp, psr, rpsr
from rl_rpsr.serializer import IntentsSerializer, TestsSerializer, VF_Serializer


def main_test(args):
    logger = logging.getLogger(__name__)
    logger.info('rl-psr-vi-test with args %s', args)

    pomdp_model = pomdp.POMDP_Model.make(args.pomdp)

    models = {}
    models['bsr'] = bsr.BSR_Model(pomdp_model)

    if args.load_core_psr is not None:
        Q = TestsSerializer().load(args.load_core_psr)
        models['psr'] = psr.PSR_Model(pomdp_model, Q)

    if args.load_core_rpsr is not None:
        I = IntentsSerializer().load(args.load_core_rpsr)
        models['rpsr'] = rpsr.RPSR_Model(pomdp_model, I)

    serializer = VF_Serializer()

    vfs = {}
    if args.load_vf_bsr is not None:
        vfs['bsr'] = serializer.load(args.load_vf_bsr)

    if args.load_vf_psr is not None:
        vfs['psr'] = serializer.load(args.load_vf_psr)

    if args.load_vf_rpsr is not None:
        vfs['rpsr'] = serializer.load(args.load_vf_rpsr)

    for i in range(args.num_tests):
        print('---------')
        print(f'TEST {i}')
        print('---------')

        # belief = models['bsr'].start
        belief = np.random.dirichlet(np.ones(pomdp_model.state_space.n))

        # print(f'BSR state {belief}')
        value = vfs['bsr'].value(belief)
        action = vfs['bsr'].policy(belief)
        print(f'BSR action {action} value {value}')

        state = models['psr'].psr(belief)
        # print(f'PSR state {state}')
        value = vfs['psr'].value(state)
        action = vfs['psr'].policy(state)
        print(f'PSR action {action} value {value}')

        state = models['rpsr'].rpsr(belief)
        # print(f'RPSR state {state}')
        value = vfs['rpsr'].value(state)
        action = vfs['rpsr'].policy(state)
        print(f'RPSR action {action} value {value}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')

    parser.add_argument('--load-core-psr', default=None)
    parser.add_argument('--load-core-rpsr', default=None)
    parser.add_argument('--load-vf-bsr', default=None)
    parser.add_argument('--load-vf-psr', default=None)
    parser.add_argument('--load-vf-rpsr', default=None)
    parser.add_argument('--num-tests', type=int, default=10)

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

    try:
        main_test(args)
    except:
        logger = logging.getLogger(__name__)
        logger.exception('The program raised an uncaught exception')
        raise


if __name__ == '__main__':
    main()
