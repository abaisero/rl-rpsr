#!/usr/bin/env python
# encoding: utf-8
from rl_rpsr import __version__
from setuptools import setup

setup(
    name='rl-rpsr',
    version=__version__,
    description='reinforcement learning with predictive state representations',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/rl-rpsr',
    packages=['rl_rpsr'],
    test_suite='tests',
    scripts=[
        'scripts/rl-rpsr-search.py',
        'scripts/rl-rpsr-info.py',
        'scripts/rl-rpsr-vi.py',
        'scripts/rl-rpsr-vi-plot.py',
        'scripts/rl-rpsr-vi-test.py',
        'scripts/rl-rpsr-sim.py',
        'scripts/rl-rpsr-eval.py',
    ],
    license='MIT',
)
