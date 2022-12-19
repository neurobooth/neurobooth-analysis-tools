from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='neurobooth-analysis-tools',
    version='0.1',
    description='Tools for loading and analyzing neurobooth data.',
    long_description=readme(),
    url='https://github.com/neurobooth/neurobooth-analysis-tools',
    author='Neurobooth Team',
    author_email='boubre@mgh.harvard.edu',
    license='BSD 3-Clause License',
    packages=['neurobooth_analysis_tools'],
    include_package_data=True,
    entrypoints={
        'console_scripts': ['slice=neurobooth-analysis-tools.slice:main']
    }
)
