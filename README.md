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

Run following examples code in `pyspark` shell:

### Cross validation example

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

### Cross validation with feature engineering example

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from joblibspark import register_spark

register_spark() # register spark backend


X_digits, y_digits = load_digits(return_X_y=True)
pca1 = PCA()
svm1 = SVC()
pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])

with parallel_backend('spark', n_jobs=3):
  scores = cross_val_score(pipe, X_digits, y_digits, cv=5)

print(scores)
```
