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
import sys
import logging
import warnings
from multiprocessing.pool import ThreadPool
import uuid
from distutils.version import LooseVersion

from joblib.parallel \
    import AutoBatchingMixin, ParallelBackendBase, register_parallel_backend, SequentialBackend
from joblib._parallel_backends import SafeFunction

import pyspark
from pyspark.sql import SparkSession
from pyspark import cloudpickle
from pyspark.util import VersionUtils


def _get_logger(name):
    """ Gets a logger by name, or creates and configures it for the first time. """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # If the logger is configured, skip the configure
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    return logger


logger = _get_logger("joblib-spark")


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
    # Hard cap on the number of concurrent hyperopt tasks (Spark jobs) to run. Set at 128.
    MAX_CONCURRENT_JOBS_ALLOWED = 128

    def __init__(self, **backend_args):
        super(SparkDistributedBackend, self).__init__(**backend_args)
        self._pool = None
        self._n_jobs = None
        self._spark = SparkSession \
            .builder \
            .appName("JoblibSparkBackend") \
            .getOrCreate()
        self._job_group = "joblib-spark-job-group-" + str(uuid.uuid4())

    def _cancel_all_jobs(self):
        if VersionUtils.majorMinorVersion(pyspark.__version__)[0] < 3:
            # Note: There's bug existing in `sparkContext.cancelJobGroup`.
            # See https://issues.apache.org/jira/browse/SPARK-31549
            warnings.warn("For spark version < 3, pyspark cancelling job API has bugs, "
                          "so we could not terminate running spark jobs correctly. "
                          "See https://issues.apache.org/jira/browse/SPARK-31549 for reference.")
        else:
            self._spark.sparkContext.cancelJobGroup(self._job_group)

    @staticmethod
    def _decide_parallelism(
        requested_parallelism, spark_default_parallelism, max_num_concurrent_tasks
    ):
        """
        Given the requested parallelism, return the max parallelism SparkTrials will actually use.
        See the docstring for `parallelism` in the constructor for expected behavior.
        """
        if max_num_concurrent_tasks == 0:
            logger.warning(
                "The cluster has no executors currently. "
                "The trials won't start until some new executors register."
            )

        if requested_parallelism is None or requested_parallelism <= 0:
            parallelism = max(spark_default_parallelism, max_num_concurrent_tasks, 1)
            logger.warning(
                "Because the requested parallelism was None or a non-positive value, "
                "parallelism will be set to ({d}), which is Spark's default parallelism ({s}), "
                "or the current total of Spark task slots ({t}), or 1, whichever is greater. "
                "We recommend setting parallelism explicitly to a positive value because "
                "the total of Spark task slots is subject to cluster sizing.".format(
                    d=parallelism,
                    s=spark_default_parallelism,
                    t=max_num_concurrent_tasks,
                )
            )
        else:
            parallelism = requested_parallelism

        if parallelism > SparkDistributedBackend.MAX_CONCURRENT_JOBS_ALLOWED:
            logger.warning(
                "Parallelism ({p}) is capped at SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED ({c}).".format(
                    p=parallelism, c=SparkDistributedBackend.MAX_CONCURRENT_JOBS_ALLOWED
                )
            )
            parallelism = SparkDistributedBackend.MAX_CONCURRENT_JOBS_ALLOWED

        if parallelism > max_num_concurrent_tasks:
            logger.warning(
                "Parallelism ({p}) is greater than the current total of Spark task slots ({c}). "
                "If dynamic allocation is enabled, you might see more executors allocated.".format(
                    p=requested_parallelism, c=max_num_concurrent_tasks
                )
            )
        return parallelism

    def effective_n_jobs(self, n_jobs):
        max_num_concurrent_tasks = self._get_max_num_concurrent_tasks()
        spark_default_parallelism = self._spark_context.defaultParallelism
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            # n_jobs=-1 means requesting all available workers
            # But if cluster in dynamic allocation mode and available workers is zero
            # then use spark_default_parallelism and trigger spark worker dynamic allocation
            n_jobs = self._decide_parallelism(
                requested_parallelism=n_jobs,
                spark_default_parallelism=spark_default_parallelism,
                max_num_concurrent_tasks=max_num_concurrent_tasks,
            )
        return n_jobs

    def _get_max_num_concurrent_tasks(self):
        # maxNumConcurrentTasks() is a package private API
        # pylint: disable=W0212
        return self._spark.sparkContext._jsc.sc().maxNumConcurrentTasks()

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
            rdd = self._spark.sparkContext.parallelize([0], 1) \
                .map(lambda _: cloudpickle.dumps(func()))
            if VersionUtils.majorMinorVersion(pyspark.__version__)[0] < 3:
                ser_res = rdd.collect()[0]
            else:
                ser_res = rdd.collectWithJobGroup(self._job_group, "joblib spark jobs")[0]
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
