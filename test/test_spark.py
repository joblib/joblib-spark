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
import os
from packaging.version import Version, parse
import sklearn
import unittest

if parse(sklearn.__version__) < Version('0.21'):
    raise RuntimeError("Test requires sklearn version >=0.21")
else:
    from joblib.parallel import Parallel, delayed, parallel_backend

from joblibspark import register_spark

from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm

from pyspark.sql import SparkSession
import pyspark

register_spark()


class TestSparkCluster(unittest.TestCase):
    spark = None

    @classmethod
    def setup_class(cls):
        cls.spark = (
            SparkSession.builder.master("local-cluster[1, 2, 1024]")
                .config("spark.task.cpus", "1")
                .config("spark.task.maxFailures", "1")
                .getOrCreate()
        )

    @classmethod
    def teardown_class(cls):
        cls.spark.stop()

    def test_simple(self):
        def inc(x):
            return x + 1

        def slow_raise_value_error(condition, duration=0.05):
            sleep(duration)
            if condition:
                raise ValueError("condition evaluated to True")

        with parallel_backend('spark') as (ba, _):
            seq = Parallel(n_jobs=5)(delayed(inc)(i) for i in range(10))
            assert seq == [inc(i) for i in range(10)]

        with pytest.raises(BaseException):
            Parallel(n_jobs=5)(delayed(slow_raise_value_error)(i == 3)
                               for i in range(10))

    def test_sklearn_cv(self):
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

    def test_job_cancelling(self):
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


@unittest.skipIf(Version(pyspark.__version__).release < (3, 4, 0),
                 "Resource group is only supported since spark 3.4.0")
class TestGPUSparkCluster(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        gpu_discovery_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "discover_2_gpu.sh"
        )

        cls.spark = (
            SparkSession.builder.master("local-cluster[1, 2, 1024]")
            .config("spark.task.cpus", "1")
            .config("spark.task.resource.gpu.amount", "1")
            .config("spark.executor.cores", "2")
            .config("spark.worker.resource.gpu.amount", "2")
            .config("spark.executor.resource.gpu.amount", "2")
            .config("spark.task.maxFailures", "1")
            .config(
                "spark.worker.resource.gpu.discoveryScript", gpu_discovery_script_path
            )
            .getOrCreate()
        )

    @classmethod
    def teardown_class(cls):
        cls.spark.stop()

    def test_resource_group(self):
        def get_spark_context(x):
            from pyspark import TaskContext
            taskcontext = TaskContext.get()
            assert taskcontext.cpus() == 1
            assert len(taskcontext.resources().get("gpu").addresses) == 1
            return TaskContext.get()

        with parallel_backend('spark') as (ba, _):
            Parallel(n_jobs=5)(delayed(get_spark_context)(i) for i in range(10))

    def test_customized_resource_group(self):
        def get_spark_context(x):
            from pyspark import TaskContext
            taskcontext = TaskContext.get()
            assert taskcontext.cpus() == 2
            assert len(taskcontext.resources().get("gpu").addresses) == 2
            return taskcontext.cpus()

        with parallel_backend('spark',
                              num_cpus_per_spark_task=2,
                              num_gpus_per_spark_task=2) as (ba, _):
            Parallel(n_jobs=5)(delayed(get_spark_context)(i) for i in range(10))
