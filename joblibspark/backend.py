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
"""
The joblib spark backend implementation.
"""
import warnings
from multiprocessing.pool import ThreadPool
import uuid
from distutils.version import LooseVersion
import os

from joblib.parallel \
    import AutoBatchingMixin, ParallelBackendBase, register_parallel_backend, SequentialBackend
from joblib._parallel_backends import SafeFunction

from pyspark.sql import SparkSession
from pyspark import cloudpickle


def register():
    """
    Register joblib spark backend.
    """
    try:
        import sklearn  # pylint: disable=C0415
        if LooseVersion(sklearn.__version__) < LooseVersion('0.21'):
            warnings.warn("Your sklearn version is < 0.21, but joblib-spark only support "
                          "sklearn >=0.21 . You can upgrade sklearn to version >= 0.21 to "
                          "make sklearn use spark backend.")
    except ImportError:
        pass
    register_parallel_backend('spark', SparkDistributedBackend)


class SparkDistributedBackend(ParallelBackendBase, AutoBatchingMixin):
    """A ParallelBackend which will execute all batches on spark.

    This backend will launch one spark job for task batch. Multiple spark job will run parallelly.
    The maximum parallelism won't exceed `sparkContext._jsc.sc().maxNumConcurrentTasks()`

    Each task batch will be run inside one spark task on worker node, and will be executed
    by `SequentialBackend`
    """

    def __init__(self, **backend_args):
        super(SparkDistributedBackend, self).__init__(**backend_args)
        self._pool = None
        self._n_jobs = None
        self._spark = SparkSession \
            .builder \
            .appName("JoblibSparkBackend") \
            .getOrCreate()
        self._job_group = "joblib-spark-job-group-" + str(uuid.uuid4())

    def _check_pyspark_pin_thread_mode(self):
        if LooseVersion(self._spark.sparkContext.version) < LooseVersion('3.0.0'):
            return False
        if os.environ.get("PYSPARK_PIN_THREAD", "false").lower() == "true":
            return True
        return False

    def _cancel_all_jobs(self):
        if self._check_pyspark_pin_thread_mode():
            self._spark.sparkContext.cancelJobGroup(self._job_group)
        else:
            # Note: There's bug existing in `sparkContext.cancelJobGroup`.
            # See https://github.com/apache/spark/pull/24898
            warnings.warn("Because pyspark py4j is not in pinned thread mode, "
                          "we could not terminate running spark jobs correctly.")

    def effective_n_jobs(self, n_jobs):
        # maxNumConcurrentTasks() is a package private API
        # pylint: disable=W0212
        max_num_concurrent_tasks = self._spark.sparkContext._jsc.sc().maxNumConcurrentTasks()
        if n_jobs == -1:
            # n_jobs=-1 means requesting all available workers
            n_jobs = max_num_concurrent_tasks
        if n_jobs > max_num_concurrent_tasks:
            n_jobs = max_num_concurrent_tasks
            warnings.warn("limit n_jobs to be maxNumConcurrentTasks in spark: " + str(n_jobs))
        return n_jobs

    def abort_everything(self, ensure_ready=True):
        self._cancel_all_jobs()
        if ensure_ready:
            self.configure(n_jobs=self.parallel.n_jobs, parallel=self.parallel,
                           **self.parallel._backend_args)  # pylint: disable=W0212

    def terminate(self):
        self._cancel_all_jobs()

    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, **backend_args):
        n_jobs = self.effective_n_jobs(n_jobs)
        self._n_jobs = n_jobs
        return n_jobs

    def _get_pool(self):
        """Lazily initialize the thread pool
        The actual pool of worker threads is only initialized at the first
        call to apply_async.
        """
        if self._pool is None:
            self._pool = ThreadPool(self._n_jobs)
        return self._pool

    def apply_async(self, func, callback=None):
        # Note the `func` args is a batch here. (BatchedCalls type)
        # See joblib.parallel.Parallel._dispatch
        def run_on_worker_and_fetch_result():
            # TODO: handle possible spark exception here. # pylint: disable=fixme
            self._spark.sparkContext.setJobGroup(self._job_group, "joblib spark jobs")
            ser_res = self._spark.sparkContext.parallelize([0], 1) \
                .map(lambda _: cloudpickle.dumps(func())) \
                .first()
            return cloudpickle.loads(ser_res)

        return self._get_pool().apply_async(
            SafeFunction(run_on_worker_and_fetch_result),
            callback=callback
        )

    def get_nested_backend(self):
        """Backend instance to be used by nested Parallel calls.
           For nested backend, always use `SequentialBackend`
        """
        nesting_level = getattr(self, 'nesting_level', 0) + 1
        return SequentialBackend(nesting_level=nesting_level), None
