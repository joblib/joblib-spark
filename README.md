# Joblib spark backend

This is joblib spark backend.

## A Note About Dependency

You need joblib >= 0.14
If you want slearn to use spark backend, you need upgrade sklearn version to >= 0.21

You need install pyspark first. Joblib-spark support spark version >= 2.4.4



## Installation

### prerequisite

1. Install python library:
```bash
pip install scikit-learn==0.21.3
pip install joblib==0.14.0
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
