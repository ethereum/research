# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ecpoly',
    version='1.0.0',
    description='Erasure code utilities for prime fields',
    long_description=readme,
    author='Vitalik Buterin',
    author_email='',
    url='https://github.com/ethereum/research/tree/master/erasure_code/ecpoly',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
    ],
)
