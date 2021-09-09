#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from time import sleep
import pytest
from distutils.version import LooseVersion
import sklearn

if LooseVersion(sklearn.__version__) < LooseVersion('0.21'):
    raise RuntimeError("Test requires sklearn version >=0.21")
else:
    from joblib.parallel import Parallel, delayed, parallel_backend

from joblibspark import register_spark

from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm

register_spark()


def inc(x):
    return x + 1


def slow_raise_value_error(condition, duration=0.05):
    sleep(duration)
    if condition:
        raise ValueError("condition evaluated to True")


def test_simple():
    with parallel_backend('spark') as (ba, _):
        seq = Parallel(n_jobs=5)(delayed(inc)(i) for i in range(10))
        assert seq == [inc(i) for i in range(10)]

    with pytest.raises(BaseException):
        Parallel(n_jobs=5)(delayed(slow_raise_value_error)(i == 3)
                           for i in range(10))


def test_sklearn_cv():
    iris = datasets.load_iris()
    clf = svm.SVC(kernel='linear', C=1)
    with parallel_backend('spark', n_jobs=3):
        scores = cross_val_score(clf, iris.data, iris.target, cv=5)

    expected = [0.97, 1.0, 0.97, 0.97, 1.0]

    for i in range(5):
        assert(pytest.approx(scores[i], 0.01) == expected[i])

    # test with default n_jobs=-1
    with parallel_backend('spark'):
        scores = cross_val_score(clf, iris.data, iris.target, cv=5)

    for i in range(5):
        assert(pytest.approx(scores[i], 0.01) == expected[i])


def test_job_cancelling():
    from joblib import Parallel, delayed
    import time
    import tempfile
    import os

    tmp_dir = tempfile.mkdtemp()

    def test_fn(x):
        if x == 0:
            # make the task-0 fail, then it will cause task 1/2/3 to be canceled.
            raise RuntimeError()
        else:
            time.sleep(15)
            # if the task finished successfully, it will write a flag file to tmp dir.
            with open(os.path.join(tmp_dir, str(x)), 'w'):
                pass

    with pytest.raises(Exception):
        with parallel_backend('spark', n_jobs=2):
            Parallel()(delayed(test_fn)(i) for i in range(2))

    time.sleep(30)  # wait until we can ensure all task finish or cancelled.
    # assert all jobs was cancelled, no flag file will be written to tmp dir.
    assert len(os.listdir(tmp_dir)) == 0
