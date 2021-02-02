import os
from setuptools import find_packages, setup

ver_file = os.path.join('research', '_version.py')
with open(ver_file) as f:
    exec(f.read())

with open("requirements.txt") as reqs:
    REQUIREMENTS = [reqs.readlines()]

with open("requirements.dev.txt") as dev_reqs:
    REQUIREMENTS_DEV = [dev_reqs.readlines()]

MAINTAINER='J. Fonseca'
MAINTAINER_EMAIL='jpfonseca@novaims.unl.pt'
VERSION = __version__
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
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']

setup(
    name='research',
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    version=VERSION,
    description='Source code for my own research. Contains most of the source code (LaTeX, Python, etc.) of all papers I have been involved in.',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    extras_require={'dev': REQUIREMENTS_DEV}
)
