import os
import codecs
from setuptools import find_packages, setup

ver_file = os.path.join('research', '_version.py')
with open(ver_file) as f:
    exec(f.read())

with open("requirements.txt") as reqs:
    REQUIREMENTS = [reqs.readlines()]

with open("requirements.dev.txt") as dev_reqs:
    REQUIREMENTS_DEV = [dev_reqs.readlines()]

MAINTAINER = 'J. Fonseca'
MAINTAINER_EMAIL = 'jpfonseca@novaims.unl.pt'
URL = 'https://github.com/joaopfonseca/ml-research'
VERSION = __version__
SHORT_DESCRIPTION = 'Implementation of Machine Learning algorithms, experiments and utilities.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LICENSE = 'MIT'
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9']

setup(
    name='ml-research',
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    download_url=URL,
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    extras_require={'dev': REQUIREMENTS_DEV}
)
