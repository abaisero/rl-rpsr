#!/usr/bin/env python
import argparse
import logging

from rl_psr import psr, rpsr
from rl_psr.pomdp import POMDP_Model
from rl_psr.serializer import CoreSerializer
from rl_psr.util import SearchType


def main_search(args):
    logger = logging.getLogger(__name__)
    logger.info('rl-psr-search with args %s', args)

    pomdp_model = POMDP_Model.make(args.pomdp)

    print(f'{args.pomdp} {args.model}', end=' ')
    print(f'|S|={pomdp_model.state_space.n}', end=' ')

    if args.model == 'psr':
        searcher = psr.searcher_factory(args.search_type)
        core = searcher.search(pomdp_model)
        print(f'|Q|={len(core)}')

    elif args.model == 'rpsr':
        searcher = rpsr.searcher_factory(args.search_type)
        core = searcher.search(pomdp_model)
        print(f'|I|={len(core)}')

    if args.save_core is not None:
        CoreSerializer().dump(args.save_core, core)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pomdp')
    parser.add_argument('model', choices=['psr', 'rpsr'])
    parser.add_argument('--save-core', default=None)
    parser.add_argument(
        '--search-type',
        type=SearchType.__getitem__,
        choices=SearchType.__members__.values(),
        default='BFS',
    )

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

    main_search(args)


if __name__ == '__main__':
    main()
