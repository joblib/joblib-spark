# Joblib spark backend

This is joblib spark backend.

## A Note About Dependency

If you installed sklearn version >= 0.21.x, then you need install a separate `joblib` libray, otherwise don't need it.

## Installation

### prerequisite

1. Install python library:
```bash
pip install cloudpickle scikit-learn
pip install joblib # This is only required when sklearn version >= 0.21.x
```

2. Install pyspark

### Install joblib-spark
```bash
cd path/to/joblib-spark
python setup.py install
```

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
