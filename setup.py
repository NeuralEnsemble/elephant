# -*- coding: utf-8 -*-

from setuptools import setup

long_description = open("README.rst").read()
install_requires = ['neo>0.3.3',
                    'numpy>=1.6.2',
                    'quantities>=0.9.0',
                    'scipy>=0.11.0']
extras_require = {'pandas': ['pandas>=0.14.0'],
                  'docs': ['numpydoc>=0.5',
                           'sphinx>=1.2.2'],
                  'tests': ['nose>=1.3.3']}

setup(
    name="elephant",
    version='0.2.0',
    packages=['elephant', 'elephant.test'],
    install_requires=install_requires,
    extras_require=extras_require,

    author="Elephant authors and contributors",
    author_email="andrew.davison@unic.cnrs-gif.fr",
    description="Elephant is a package for analysis of electrophysiology data in Python",
    long_description=long_description,
    license="BSD",
    url='http://neuralensemble.org/elephant',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering']
)
