import os

from setuptools import setup, find_packages

# Read the contents of the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()


def get_version():
    version_file = os.path.normpath(os.path.join(os.path.dirname(__file__), 'VERSION'))
    with open(version_file) as fh:
        version = fh.read().strip()
        return version


setup(
    name='orbuculum',
    version=get_version(),
    description='Orbuculum is a tool to ask questions based on provide PDF documents.',
    author='Kent Huang',
    author_email='kent@infuseai.io',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'orbuculum = orbuculum.cli:cli',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
)
