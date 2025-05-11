from setuptools import setup, find_packages

setup(
    name='BayesMLR',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'time',
        'matplotlib',
    ],
    description='A package implements the fast Bayesian linear regression',
    author='Jian Kang',
)