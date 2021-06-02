#!/usr/bin/env python
import argparse
import logging
from dataclasses import dataclass, field
from typing import List

from rl_psr import bsr, pomdp, psr, rpsr
from rl_psr.policy import ModelPolicy, Policy, RandomPolicy
from rl_psr.serializer import IntentsSerializer, TestsSerializer, VF_Serializer


def make_policy(models, pomdp_model, args) -> Policy:
    if args.policy == 'random':
        policy = RandomPolicy(pomdp_model)

    else:
        serializer = VF_Serializer()

        if args.policy == 'bsr':
            vf = serializer.load(args.load_vf_bsr)
        elif args.policy == 'psr':
            vf = serializer.load(args.load_vf_psr)
        elif args.policy == 'rpsr':
            vf = serializer.load(args.load_vf_rpsr)

        model = models[args.policy]
        policy = ModelPolicy(model, vf)

    return policy


def return_(rewards, discount):
    G, d = 0.0, 1.0
    for r in rewards:
        G += r * d
        d *= discount

    return G


@dataclass
class Simulation:
    actions: List[int] = field(default_factory=list)
    rewards: List[int] = field(default_factory=list)
    observations: List[int] = field(default_factory=list)

    def append(self, action, reward, observation):
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations.append(observation)


def simulate(
    env, policy, num_steps
) -> Simulation:  # pylint: disable=too-many-locals

    logger = logging.getLogger(__name__)

    sim = Simulation()

    env.reset()
    action = policy.reset()
    for _ in range(num_steps - 1):
        _, reward, _, info = env.step(action)
        observation = info['observation']

        logger.info(
            'action %s observation %s reward %f',
            env.model.actions[action],
            env.model.observations[observation],
            reward,
        )

        sim.append(action, reward.item(), observation)

        action = policy.step(action, observation)

    return sim


def model_rewards(model, actions, observations):
    rewards = []

    state = model.start.copy()
    for action, observation in zip(actions, observations):
        rewards.append(model.expected_reward(state, action))
        state = model.dynamics(state, action, observation)

    return rewards


def main_eval(args):
    logger = logging.getLogger(__name__)
    logger.info('rl-psr-eval with args %s', args)

    pomdp_model = pomdp.POMDP_Model.make(args.pomdp)

    models = {}
    models['bsr'] = bsr.BSR_Model(pomdp_model)

    if args.load_core_psr is not None:
        Q = TestsSerializer().load(args.load_core_psr)
        models['psr'] = psr.PSR_Model(pomdp_model, Q)

    if args.load_core_rpsr is not None:
        I = IntentsSerializer().load(args.load_core_rpsr)
        models['rpsr'] = rpsr.RPSR_Model(pomdp_model, I)

    if args.env == 'bsr':
        env = bsr.BSR(models['bsr'])

    elif args.env == 'psr':
        env = psr.PSR(models['psr'])

    elif args.env == 'rpsr':
        env = rpsr.RPSR(models['rpsr'])

    policy = make_policy(models, pomdp_model, args)

    logger.info('pomdp %s env %s policy %s', args.pomdp, args.env, args.policy)
    for i in range(args.num_simulations):
        logger.info('simulation %d / %d', i, args.num_simulations)
        sim = simulate(env, policy, num_steps=args.num_steps)

        for key, model in models.items():
            rewards = model_rewards(model, sim.actions, sim.observations)
            s = f'{args.env} {args.policy} {key} {return_(rewards, env.discount)}'
            logger.info(s)
            print(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')
    parser.add_argument('env', choices=['bsr', 'psr', 'rpsr'])
    parser.add_argument(
        'policy', choices=['random', 'bsr', 'psr', 'rpsr'], default='random'
    )

    parser.add_argument('--load-core-psr', default=None)
    parser.add_argument('--load-core-rpsr', default=None)
    parser.add_argument('--load-vf-bsr', default=None)
    parser.add_argument('--load-vf-psr', default=None)
    parser.add_argument('--load-vf-rpsr', default=None)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--num-simulations', type=int, default=1)

    parser.add_argument('--log-filename', default=None)
    parser.add_argument(
        '--log-level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        default='INFO',
    )

    args = parser.parse_args()

    if args.env == 'psr' and args.load_core_psr is None:
        parser.error(
            'argument --env: invalid choice: \'psr\' (psr model not loaded)'
        )

    if args.env == 'rpsr' and args.load_core_rpsr is None:
        parser.error(
            'argument --env: invalid choice: \'rpsr\' (rpsr model not loaded)'
        )

    if args.policy == 'psr' and args.load_core_psr is None:
        parser.error(
            'argument --policy: invalid choice: \'psr\' (psr model not loaded)'
        )

    if args.policy == 'rpsr' and args.load_core_rpsr is None:
        parser.error(
            'argument --policy: invalid choice: \'rpsr\' (rpsr model not loaded)'
        )

    if args.policy == 'bsr' and args.load_vf_bsr is None:
        parser.error(
            'argument --policy: invalid choice: \'bsr\' (bsr vf not loaded)'
        )

    if args.policy == 'psr' and args.load_vf_psr is None:
        parser.error(
            'argument --policy: invalid choice: \'psr\' (psr vf not loaded)'
        )

    if args.policy == 'rpsr' and args.load_vf_rpsr is None:
        parser.error(
            'argument --policy: invalid choice: \'rpsr\' (rpsr vf not loaded)'
        )

    if args.log_filename is not None:
        logging.basicConfig(
            filename=args.log_filename,
            datefmt='%Y/%m/%d %H:%M:%S',
            format='%(asctime)s %(relativeCreated)d %(levelname)-8s %(name)-12s %(funcName)s - %(message)s',
            level=getattr(logging, args.log_level),
        )

    try:
        main_eval(args)
    except:
        logger = logging.getLogger(__name__)
        logger.exception('The program raised an uncaught exception')
        raise


if __name__ == '__main__':
    main()
