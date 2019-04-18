# -*- coding: utf-8 -*-

from setuptools import setup
import os
import sys
import platform
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

long_description = open("README.rst").read()
with open('requirements.txt', 'r') as fp:
    install_requires = fp.read()
extras_require = {}
for extra in ['extras', 'docs', 'tests']:
    with open('requirements-{0}.txt'.format(extra), 'r') as fp:
        extras_require[extra] = fp.read()

# spade specific
is_64bit = sys.maxsize > 2 ** 32
is_python3 = float(sys.version[0:3]) > 2.7
if platform.uname()[0] == "Windows":
    oext = ".pyd"
elif platform.uname()[0] == "Linux":
    oext = ".so"
else:
    oext = None

if oext:
    if is_python3:
        py_ver = '3'
    else:
        py_ver = '2'
    if is_64bit:
        arch = '64'
    else:
        arch = '32'

    try:
        urlretrieve('http://www.borgelt.net/bin' +
                    arch + '/py' + py_ver + '/fim' + oext,
                    'elephant/spade_src/fim' + oext)
    except:
        print("Unable to download fim" + oext + " module.")

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
        os.path.join('spade_src', '*.pyd')
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
