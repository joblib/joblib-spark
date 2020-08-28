# Joblib Apache Spark Backend

This library provides Apache Spark backend for joblib to distribute tasks on a Spark cluster.

## Installation

`joblibspark` requires Python 3.6+, `joblib>=0.14` and `pyspark>=2.4` to run.
To install `joblibspark`, run:

```bash
pip install joblibspark
```

The installation does not install PySpark because for most users, PySpark is already installed.
If you do not have PySpark installed, you can install `pyspark` together with `joblibspark`:

```bash
pip install pyspark>=3.0.0 joblibspark
```

If you want to use `joblibspark` with `scikit-learn`, please install `scikit-learn>=0.21`.

## Examples

Run following example code in `pyspark` shell:

```python
from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from joblibspark import register_spark

register_spark() # register spark backend

iris = datasets.load_iris()
clf = svm.SVC(kernel='linear', C=1)
with parallel_backend('spark', n_jobs=3):
  scores = cross_val_score(clf, iris.data, iris.target, cv=5)

print(scores)
```

## Limitations

`joblibspark` does not generally support run model inference and feature engineering in parallel.
For example:

```python
from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=10)
with parallel_backend('spark', n_jobs=3):
    # This won't run parallelly on spark, it will still run locally.
    h.transform(...)

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

with parallel_backend('spark', n_jobs=3):
    # This won't run parallelly on spark, it will still run locally.
    regr.predict(diabetes_X_test)
```

Note: for `sklearn.ensemble.RandomForestClassifier`, there is a `n_jobs` parameter,
that means the algorithm support model training/inference in parallel,
but in its inference implementation, it bind the backend to built-in backends,
so the spark backend not work for this case.
