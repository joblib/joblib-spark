Installing joblib
===================

Using `pip`
------------

`joblibspark` requires Python 3.6+, `joblib>=0.14` and `pyspark>=2.4.0` to run.
* Note: For spark version < 3, pyspark cancelling job API has bugs, so we could
  not terminate running spark jobs correctly. See
  https://issues.apache.org/jira/browse/SPARK-31549 for reference.
  So we recommend to install spark 3.0 which addressed this issue.

* To install `joblibspark`, run::

    pip install joblibspark

  You may need to run the above command as administrator

  On a unix environment, it is better to install outside of the hierarchy
  managed by the system::

    pip install --prefix /usr/local joblibspark

  Installing only for a specific user is easy if you use Python 2.7 or
  above::

    pip install --user joblibspark

* The installation above does not install PySpark because for most users,
  PySpark is already installed. If you do not have PySpark installed, you can
  install `pyspark` together with `joblibspark`::

    pip install pyspark>=3.0.0 joblibspark

* If you want to use `joblibspark` with `scikit-learn`, please install `scikit-learn>=0.21`::

    pip install scikit-learn>=0.21

Using distributions
--------------------

Joblibspark is packaged for several linux distribution: archlinux, debian,
ubuntu, altlinux, and fedora. For minimum administration overhead, using the
package manager is the recommended installation strategy on these
systems.

The manual way
---------------

To install joblibspark first download the latest tarball (follow the link on
the bottom of http://pypi.python.org/pypi/joblib) and expand it.

Installing in a local environment
..................................

If you don't need to install for all users, we strongly suggest that you
create a local environment and install `joblibspark` in it. One of the pros of
this method is that you never have to become administrator, and thus all
the changes are local to your account and easy to clean up.
Simply move to the directory created by expanding the `joblibspark` tarball
and run the following command::

    python setup.py install --user

Installing for all users
........................

If you have administrator rights and want to install for all users, all
you need to do is to go in directory created by expanding the `joblibspark`
tarball and run the following line::

    python setup.py install

If you are under Unix, we suggest that you install in '/usr/local' in
order not to interfere with your system::

    python setup.py install --prefix /usr/local
