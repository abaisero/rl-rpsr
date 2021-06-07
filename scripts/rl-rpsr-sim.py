#!/usr/bin/env python
import argparse
import logging

from rl_rpsr import bsr, psr, rpsr
from rl_rpsr.pomdp import POMDP_Model
from rl_rpsr.serializer import IntentsSerializer, TestsSerializer, VF_Serializer


def make_random_policy(env):
    def policy(state):  # pylint: disable=unused-argument
        return env.action_space.sample()

    return policy


def main_sim(args):
    pomdp_model = POMDP_Model.make(args.pomdp)

    if args.model == 'bsr':
        model = bsr.BSR_Model(pomdp_model)
        env = bsr.BSR(model)

    elif args.model == 'psr':
        Q = TestsSerializer().load(args.load_core)
        model = psr.PSR_Model(pomdp_model, Q)
        env = psr.PSR(model)

    elif args.model == 'rpsr':
        I = IntentsSerializer().load(args.load_core)
        model = rpsr.RPSR_Model(pomdp_model, I)
        env = rpsr.RPSR(model)

    if args.load_vf is None:
        policy = make_random_policy(env)
    else:
        vf = VF_Serializer().load(args.load_vf)
        policy = vf.policy

    state = env.reset()
    for _ in range(args.num_steps):
        action = policy(state)
        state, reward, _, info = env.step(action)
        observation = info['observation']

        print(
            f'{env.model.actions[action]:10} {reward: .3f} {env.model.observations[observation]:10}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')
    parser.add_argument('model', choices=['bsr', 'psr', 'rpsr'])
    parser.add_argument('--load-core', default=None)
    parser.add_argument('--load-vf', default=None)
    parser.add_argument('--num-steps', type=int, default=100)

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

    main_sim(args)


if __name__ == '__main__':
    main()
