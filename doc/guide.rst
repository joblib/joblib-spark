
.. _guide:

==========================
Joblib spark backend guide
==========================

Common usage
============

Joblib-spark provides a spark backend, before using it, we need register it first::

    >>> from joblibspark import register_spark
    >>> register_spark()

Then we can use spark backend to run joblib tasks parallelly like::

    >>> from joblib.parallel import Parallel, delayed, parallel_backend
    >>> with parallel_backend('spark') as (ba, _):
    ...     Parallel(n_jobs=5)(delayed(inc)(i) for i in range(3))
    [1, 2, 3]


Use joblib spark backend in scikit-learn
=====================================================

If you install scikit-learn with version >= 0.21, then scikit-learn can use this
spark backend::

    >>> from sklearn.utils import parallel_backend
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn import datasets
    >>> from sklearn import svm
    >>> from joblibspark import register_spark
    >>>
    >>> register_spark() # register spark backend
    >>>
    >>> iris = datasets.load_iris()
    >>> clf = svm.SVC(kernel='linear', C=1)
    >>> with parallel_backend('spark', n_jobs=3):
    ...   cross_val_score(clf, iris.data, iris.target, cv=5)
    array([0.96666667, 1.        , 0.96666667, 0.96666667, 1.        ])

