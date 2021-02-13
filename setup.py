# -*- coding: utf-8 -*-

import os
import platform
import struct
import sys
from urllib.request import urlretrieve

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__),
                       "elephant", "VERSION")) as version_file:
    version = version_file.read().strip()

with open("README.md") as f:
    long_description = f.read()
with open('requirements/requirements.txt') as fp:
    install_requires = fp.read().splitlines()
extras_require = {}
for extra in ['extras', 'docs', 'tests', 'tutorials', 'cuda', 'opencl']:
    with open('requirements/requirements-{0}.txt'.format(extra)) as fp:
        extras_require[extra] = fp.read()


def download_spade_fim():
    """
    Downloads SPADE specific PyFIM binary file.
    """
    if platform.system() == "Windows":
        fim_filename = "fim.pyd"
    else:
        # Linux
        fim_filename = "fim.so"
    spade_src_dir = os.path.join(os.path.dirname(__file__), "elephant",
                                 "spade_src")
    fim_lib_path = os.path.join(spade_src_dir, fim_filename)
    if os.path.exists(fim_lib_path):
        return

    arch = struct.calcsize("P") * 8
    py_ver = sys.version_info.major
    url_fim = f"http://www.borgelt.net/bin{arch}/py{py_ver}/{fim_filename}"
    try:
        urlretrieve(url_fim, filename=fim_lib_path)
        print("Successfully downloaded fim lib to {}".format(fim_lib_path))
    except Exception:
        print("Unable to download {url} module.".format(url=url_fim))


if len(sys.argv) > 1 and sys.argv[1].lower() != 'sdist':
    download_spade_fim()

setup(
    name="elephant",
    version=version,
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
