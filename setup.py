# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup

python_version_major = sys.version_info.major

with open("README.rst") as f:
    long_description = f.read()
with open('requirements.txt') as fp:
    install_requires = fp.read()
extras_require = {}
for extra in ['extras', 'docs', 'tests']:
    with open('requirements-{0}.txt'.format(extra)) as fp:
        extras_require[extra] = fp.read()

setup(
    name="elephant",
    version='0.6.2',
    packages=['elephant', 'elephant.test'],
    package_data={'elephant': [
        os.path.join('current_source_density_src', 'test_data.mat'),
        os.path.join('current_source_density_src', 'LICENSE'),
        os.path.join('current_source_density_src', 'README.md'),
        os.path.join('current_source_density_src', '*.py'),
        os.path.join('spade_src', '*.py'),
        os.path.join('spade_src', 'LICENSE'),
        os.path.join('spade_src', '*.so'),
        os.path.join('spade_src', '*.pyd'),
    ]},

    install_requires=install_requires,
    extras_require=extras_require,

    author="Elephant authors and contributors",
    author_email="andrew.davison@unic.cnrs-gif.fr",
    description="Elephant is a package for analysis of electrophysiology"
                " data in Python",
    long_description=long_description,
    license="BSD",
    url='http://neuralensemble.org/elephant',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering']
)
