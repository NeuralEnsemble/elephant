# -*- coding: utf-8 -*-

import os
import platform
import struct
import sys

from setuptools import setup

python_version_major = sys.version_info.major

if python_version_major == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


def download_spade_fim():
    """
    Downloads SPADE specific PyFIM binary file.
    """
    arch_bits = struct.calcsize("P") * 8
    if platform.system() == "Windows":
        fim_filename = "fim.pyd"
    else:
        # Linux
        fim_filename = "fim.so"
    url_fim = "http://www.borgelt.net/bin{arch}/py{py_ver}/{filename}". \
        format(arch=arch_bits, py_ver=python_version_major,
               filename=fim_filename)

    try:
        urlretrieve(url_fim,
                    filename=os.path.join('elephant', 'spade_src',
                                          fim_filename))
    except:
        print("Unable to download {url} module.".format(url=url_fim))


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
    version='0.6.2',
    packages=['elephant', 'elephant.test'],
    package_data={'elephant': [
        os.path.join('current_source_density_src', 'test_data.mat'),
        os.path.join('current_source_density_src', 'LICENSE'),
        os.path.join('current_source_density_src', 'README.md'),
        os.path.join('current_source_density_src', '*.py'),
        os.path.join('spade_src', '*.py'),
        os.path.join('spade_src', 'LICENSE'),
        os.path.join('test', '*.txt')
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
