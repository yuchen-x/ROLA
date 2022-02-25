#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

setup(
    name='pg_marl',
    version='0.0.1',
    description='pg_marl - policy gradient for multi-agent reinforcement learning',
    author='Yuchen Xiao',
    author_email='xiao.yuch@northeastern.edu',

    packages=['pg_marl', 'marl_envs'],
    package_dir={'': 'src'},

    scripts=[
        'scripts/rola.py',
    ],

    license='MIT',
)
