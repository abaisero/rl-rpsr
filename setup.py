#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

from rl_psr import __version__

setup(
    name='rl-psr',
    version=__version__,
    description='reinforcement learning with predictive state representations',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/rl-psr',
    packages=['rl_psr'],
    test_suite='tests',
    scripts=[
        'scripts/rl-psr-search.py',
        'scripts/rl-psr-info.py',
        'scripts/rl-psr-vi.py',
        'scripts/rl-psr-vi-plot.py',
        'scripts/rl-psr-vi-test.py',
        'scripts/rl-psr-sim.py',
        'scripts/rl-psr-eval.py',
    ],
    license='MIT',
)
