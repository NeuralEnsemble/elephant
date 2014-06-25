# -*- coding: utf-8 -*-

from setuptools import setup

long_description = open("README.rst").read()
install_requires = ['neo>=0.3.3',
                    'numpy>=1.5.0',
                    'quantities>=0.9.0',
                    'scipy'>='0.14.0']


setup(
    name = "elephant",
    version = '0.1dev',
    packages = ['elephant', 'elephant.test'],
    install_requires=install_requires,
    author = "Elephant authors and contributors",
    author_email = "andrew.davison@unic.cnrs-gif.fr",
    description = "Elephant is a package for analysis of electrophysiology data in Python",
    long_description = long_description,
    license = "BSD",
    url='http://neuralensemble.org/elephant',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering']
)
