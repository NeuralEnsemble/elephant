# -*- coding: utf-8 -*-

from setuptools import setup
import os
try:
    from distutils.extension import Extension
    from Cython.Build import cythonize
    exts = [Extension('*', ['elephant/*.pyx'], include_dirs=['elephant'])]
    exts = cythonize(exts)
except ImportError as ie:
    exts = []
    # no loop for you!
    pass


long_description = open("README.rst").read()
install_requires = ['neo>0.3.3',
                    'numpy>=1.8.2',
                    'quantities>=0.10.1',
                    'scipy>=0.14.0']
extras_require = {'pandas': ['pandas>=0.14.1'],
                  'docs': ['numpydoc>=0.5',
                           'sphinx>=1.2.2'],
                  'tests': ['nose>=1.3.3'],
                  'cython': ['cython>=0.24.1']}

<<<<<<< dev_multitapered_spectral_analysis
=======
try:
    from distutils.extension import Extension
    from Cython.Distutils import build_ext as build_pyx_ext
    from numpy import get_include
    # add Cython extensions to the setup options
    exts = [Extension('_cython_utils', ['elephant/_cython_utils.pyx'])]
except ImportError:
    build_pyx_ext = None
    exts = []
    # no loop for you!
    pass

>>>>>>> dev_multitapered_spectral_analysis

setup(
    name="elephant",
    version='0.4.1',
    packages=['elephant', 'elephant.test'],
<<<<<<< dev_multitapered_spectral_analysis
    package_data={'elephant': [os.path.join('current_source_density_src', 'test_data.mat'),
                               os.path.join(
                                   'current_source_density_src', 'LICENSE'),
                               os.path.join(
                                   'current_source_density_src', 'README.md'),
                               os.path.join('current_source_density_src', '*.py')]},
=======
    package_data = {'elephant' : [os.path.join('icsd', 'test_data.mat'),
                                  os.path.join('icsd', 'LICENSE'),
                                  os.path.join('icsd', 'README.md'),
                                  os.path.join('test', 'dpss_testdata1.txt'),
                                  os.path.join('.', '_cython_utils.pyx'),
                                  os.path.join('test', 'dpss_testdata2.npy'),
                                  ]},
>>>>>>> dev_multitapered_spectral_analysis
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
        'Topic :: Scientific/Engineering'],
    ext_modules=exts
)
