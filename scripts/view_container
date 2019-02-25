#!/usr/bin/python

import argparse
from cremi_tools.viewer.volumina import view_container


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--ndim', type=int, default=3)
    parser.add_argument('--shape', type=int, nargs='+', default=None)

    args = parser.parse_args()
    view_container(args.path, args.ndim, args.shape)
