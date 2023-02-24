"""
Manages processes involved in AlphaZero with supervised learning: Self-play, training, evaluation, and training on pro data
"""

import argparse
from logging import getLogger

_LOGGER = getLogger(__name__)
_CMD_LIST = ['self', 'eval', 'opt', 'sl']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', choices = _CMD_LIST)
    return parser

def setup():
    return NotImplemented

def start():
    parser = create_parser()
    args = parser.parse_args()

    match args.cmd:
        case 'self':
            from .workers import self_play
            print('self')
        case 'eval':
            from .workers import eval
            print('eval')
        case 'opt':
            from .workers import opt
            print('opt')
        case 'sl':
            from .workers import sl
            print('sl')