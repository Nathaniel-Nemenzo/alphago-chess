"""
Manages processes involved in AlphaZero with supervised learning: Self-play, training, evaluation, and training on pro data
"""

import argparse
from logging import getLogger

_LOGGER = getLogger(__name__)
_CMD_LIST = ['train', 'opt']

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
        case 'train':
            from .workers import train
            train.start()
        case 'opt':
            from .workers import self_play
            self_play.start()
        case _:
            print('Usage: ./run.py --cmd [train | self-play]')