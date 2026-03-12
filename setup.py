# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path

from setuptools import setup


NAME = 'demucs'
DESCRIPTION = 'Music source separation in the waveform domain.'

URL = 'https://github.com/facebookresearch/demucs'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Défossez'
REQUIRES_PYTHON = '>=3.12'

HERE = Path(__file__).parent

# Get version without explicitely loading the module.
for line in open('demucs/__init__.py', encoding='utf-8'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']


def load_requirements(name, visited=None):
    if visited is None:
        visited = set()

    path = HERE / name
    if path in visited:
        return []
    visited.add(path)

    required = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-r '):
            required.extend(load_requirements(line.split(None, 1)[1], visited))
            continue
        if line.startswith('--requirement '):
            required.extend(load_requirements(line.split(None, 1)[1], visited))
            continue
        if line.startswith('-'):
            continue
        required.append(line)
    return required


REQUIRED = load_requirements('requirements_minimal.txt')

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['demucs', 'demucs.bin'],
    extras_require={
        # GPU users: install torch from CUDA index first, then use this extra
        #   uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
        #   uv pip install -e ".[gpu,enhanced]"
        'cpu': [
            'torch>=2.0.0',
            'torchaudio>=2.0.0',
        ],
        'gpu': [
            'torch>=2.0.0',
            'torchaudio>=2.0.0',
        ],
        'enhanced': [
            'soundfile>=0.12.0',
            'av>=12.0.0',
        ],
        'dev': load_requirements('requirements_dev.txt'),
    },
    install_requires=REQUIRED,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'demucs=demucs.separate:main',
            'demucs-enhanced=demucs.bin.run_enhanced:main',
        ],
    },
    license='MIT License',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
