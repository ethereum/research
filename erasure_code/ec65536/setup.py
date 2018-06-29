# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ec65536',
    version='1.0.0',
    description='Erasure code utilities for GF16',
    long_description=readme,
    author='Vitalik Buterin',
    author_email='',
    url='https://github.com/ethereum/research/tree/master/erasure_code/ec65536',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
    ],
)
