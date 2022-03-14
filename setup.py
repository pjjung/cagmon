# -*- coding: utf-8 -*-
# Copyright (C) Phil Jung (2020)
#
# This file is port of CAGMon.
#
# CAGMon is the tool that evaluates the dependence between the primary and auxiliary channels of Gravitational-Wave detectors.
#
# CAGMon is following the GNU General Public License version 3. Under this term, you can redistribute and/or modify it.
# See the GNU free software license for more details.

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='CAGMon',
    version='0.8.5',
    author='Phil Jung',
    author_email='pjjung@nims.re.kr',
    description='Correlation Analysis based on Glitch Monitoring',
    url='https://github.com/pjjung/cagmon',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['test','examples']),
    entry_points={
        "console_scripts": [
            "cagmon = cagmon.main:main"
        ]
    },
    install_requires=["setuptools",
                      "gwpy>=1.0.1",
                      "lalsuite>=1.4.4",
                      "minepy>=1.2.5"],
    python_requires=">=2.7",
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
    ],
)
