#!/usr/bin/env python3
"""
Ominari Trading System Setup
"""

from setuptools import setup, find_packages

setup(
    name="ominari",
    version="1.0.0",
    description="Automated Sports Betting Trading System with Blockchain Integration",
    author="Ominari Team",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ominari=main:main',
        ],
    },
    python_requires='>=3.11',
)