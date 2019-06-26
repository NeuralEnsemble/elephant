# -*- coding: utf-8 -*-

import sys

from setuptools import setup

from elephant import __version__
from elephant.spade_src.fim_manager import download_spade_fim

python_version_major = sys.version_info.major

with open("README.rst") as f:
    long_description = f.read()
with open('requirements.txt') as fp:
    install_requires = fp.read()
extras_require = {}
for extra in ['extras', 'docs', 'tests']:
    with open('requirements-{0}.txt'.format(extra)) as fp:
        extras_require[extra] = fp.read()

download_spade_fim()

setup(
    name="elephant",
    version=__version__,
    packages=['elephant', 'elephant.test'],
    include_package_data=True,

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
