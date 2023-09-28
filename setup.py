"""
Setup script for adversarial-insight-ml package.

This script configures the packaging and distribution of the adversarial-insight-ml package.
If you are not involved in building/publishing package to PyPI, you do not need to run this.
"""


from setuptools import setup, find_packages

requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='adversarial-insight-ml',
    version='0.1.1',
    author='Team 7',
    description='An evaluation tool for ML models defense against adversarial attack',
    long_description=open('README.md').read(),  # Show README.md on PyPI page
    long_description_content_type='text/markdown',
    url='https://github.com/uoa-compsci399-s2-2023/capstone-project-team-7',
    project_urls={
        'Bug Tracker': 'https://github.com/uoa-compsci399-s2-2023/capstone-project-team-7/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.9',
    install_requires=requirements,
)
