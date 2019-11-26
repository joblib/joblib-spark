from setuptools import setup, find_packages

import joblibspark

if __name__ == '__main__':
    setup(name='joblibspark',
          version=joblibspark.__version__,
          author='Weichen Xu',
          author_email='weichen.xu@databricks.com',
          url='https://github.com/joblib/joblib-spark',
          description="Joblib Apache Spark Backend",
          long_description=joblibspark.__doc__,
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Education',
              'License :: OSI Approved :: Apache Software License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Topic :: Scientific/Engineering',
              'Topic :: Utilities',
              'Topic :: Software Development :: Libraries',
          ],
          platforms='any',
          packages=find_packages(),
          install_requires=[
              'joblib>=0.14',
          ])
