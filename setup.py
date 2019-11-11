#!/usr/bin/env python

from setuptools import setup
import sys

import joblibspark

# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist', 'bdist_egg', 'bdist_dumb',
            'bdist_rpm', 'bdist_wheel', 'bdist_wininst', 'install_egg_info',
            'egg_info', 'easy_install', 'upload',
            )).intersection(sys.argv)) > 0:
    import setuptools

extra_setuptools_args = {}


if __name__ == '__main__':
    setup(name='joblibspark',
          version=joblibspark.__version__,
          author='Weichen Xu',
          author_email='weichen.xu@databricks.com',
          url='https://github.com/WeichenXu123/joblib-spark',
          description="A library for joblib spark backend.",
          long_description=joblibspark.__doc__,
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Education',
              'License :: OSI Approved :: Apache Software License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Topic :: Scientific/Engineering',
              'Topic :: Utilities',
              'Topic :: Software Development :: Libraries',
          ],
          platforms='any',
          packages=setuptools.find_packages(),
          **extra_setuptools_args)
