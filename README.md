# Joblib Apache Spark Backend

This library provides Apache Spark backend for joblib to distribute tasks on a Spark cluster.

## Installation

`joblibspark` requires Python 3.6+, `joblib>=0.14` and `pyspark>=2.4.4` to run.
To install `joblibspark`, run:

```bash
pip install joblibspark
```

The installation does not install PySpark because for most users, PySpark is already installed.
If you do not have PySpark installed, you can install `pyspark` together with `joblibspark`:

```bash
pip install pyspark>=2.4.4 joblibspark
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
