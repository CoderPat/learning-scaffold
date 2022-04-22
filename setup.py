# -*- coding: utf-8 -*-
from setuptools import find_packages, setup


setup(
    name="smat",
    version="0.0.1",
    author="Patrick Fernandes & Marcos Treviso",
    author_email="pfernand@cs.cmu.edu",
    url="https://github.com/CoderPat/learning-scaffold/",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=['jax','jaxlib','datasets','optax','pkbar','wandb'],
)
