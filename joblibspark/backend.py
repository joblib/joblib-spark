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
import atexit
import warnings
from multiprocessing.pool import ThreadPool
import uuid
from typing import Optional
from packaging.version import Version, parse

from joblib.parallel \
    import AutoBatchingMixin, ParallelBackendBase, register_parallel_backend, SequentialBackend

try:
    # joblib >=1.4.0
    from joblib._utils import _TracebackCapturingWrapper as SafeFunction
except ImportError:
    try:
        from joblib._parallel_backends import SafeFunction
    except ImportError:
        # joblib >= 1.3.0
        from joblib._parallel_backends import PoolManagerMixin
        SafeFunction = None

from py4j.clientserver import ClientServer

import pyspark
from pyspark import cloudpickle
from pyspark.util import VersionUtils

from .utils import create_resource_profile, get_spark_session


def register():
    """
    Register joblib spark backend.
    """
    try:
        import sklearn  # pylint: disable=C0415
        if parse(sklearn.__version__) < Version('0.21'):
            warnings.warn("Your sklearn version is < 0.21, but joblib-spark only support "
                          "sklearn >=0.21 . You can upgrade sklearn to version >= 0.21 to "
                          "make sklearn use spark backend.")
    except ImportError:
        pass
    register_parallel_backend('spark', SparkDistributedBackend)


# pylint: disable=too-many-instance-attributes
class SparkDistributedBackend(ParallelBackendBase, AutoBatchingMixin):
    """A ParallelBackend which will execute all batches on spark.

    This backend will launch one spark job for task batch. Multiple spark job will run parallelly.
    The maximum parallelism won't exceed `sparkContext._jsc.sc().maxNumConcurrentTasks()`

    Each task batch will be run inside one spark task on worker node, and will be executed
    by `SequentialBackend`
    """

    def __init__(self,
                 num_cpus_per_spark_task: Optional[int] = None,
                 num_gpus_per_spark_task: Optional[int] = None,
                 **backend_args):
        # pylint: disable=super-with-arguments
        super(SparkDistributedBackend, self).__init__(**backend_args)
        self._pool = None
        self._n_jobs = None
        self._spark = get_spark_session()
        self._spark_context = self._spark.sparkContext
        self._job_group = "joblib-spark-job-group-" + str(uuid.uuid4())
        self._spark_pinned_threads_enabled = isinstance(
            self._spark_context._gateway, ClientServer
        )
        self._spark_supports_job_cancelling = (
            self._spark_pinned_threads_enabled
            or hasattr(self._spark_context.parallelize([1]), "collectWithJobGroup")
        )
        self._is_running = False
        try:
            from IPython import get_ipython  # pylint: disable=import-outside-toplevel
            self._ipython = get_ipython()
        except ImportError:
            self._ipython = None

        self._support_stage_scheduling = self._is_support_stage_scheduling()
        self._resource_profile = self._create_resource_profile(num_cpus_per_spark_task,
                                                               num_gpus_per_spark_task)

    def _is_support_stage_scheduling(self):
        spark_master = self._spark_context.master
        is_spark_local_mode = spark_master == "local" or spark_master.startswith("local[")
        if is_spark_local_mode:
            support_stage_scheduling = False
            warnings.warn("Spark local mode doesn't support stage-level scheduling.")
        else:
            support_stage_scheduling = hasattr(
                self._spark_context.parallelize([1]), "withResources"
            )
            warnings.warn("Spark version does not support stage-level scheduling.")
        return support_stage_scheduling

    def _create_resource_profile(self,
                                 num_cpus_per_spark_task,
                                 num_gpus_per_spark_task) -> Optional[object]:
        resource_profile = None
        if self._support_stage_scheduling:
            self.using_stage_scheduling = True

            default_cpus_per_task = int(self._spark.conf.get("spark.task.cpus", "1"))
            default_gpus_per_task = int(self._spark.conf.get("spark.task.resource.gpu.amount", "0"))
            num_cpus_per_spark_task = num_cpus_per_spark_task or default_cpus_per_task
            num_gpus_per_spark_task = num_gpus_per_spark_task or default_gpus_per_task

            resource_profile = create_resource_profile(num_cpus_per_spark_task,
                                                       num_gpus_per_spark_task)

        return resource_profile

    def _cancel_all_jobs(self):
        self._is_running = False
        if not self._spark_supports_job_cancelling:
            # Note: There's bug existing in `sparkContext.cancelJobGroup`.
            # See https://issues.apache.org/jira/browse/SPARK-31549
            warnings.warn("For spark version < 3, pyspark cancelling job API has bugs, "
                          "so we could not terminate running spark jobs correctly. "
                          "See https://issues.apache.org/jira/browse/SPARK-31549 for reference.")
        else:
            self._spark.sparkContext.cancelJobGroup(self._job_group)

    def effective_n_jobs(self, n_jobs):
        max_num_concurrent_tasks = self._get_max_num_concurrent_tasks()
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            # n_jobs=-1 means requesting all available workers
            n_jobs = max_num_concurrent_tasks
        if n_jobs > max_num_concurrent_tasks:
            warnings.warn(f"User-specified n_jobs ({n_jobs}) is greater than the max number of "
                          f"concurrent tasks ({max_num_concurrent_tasks}) this cluster can run now."
                          "If dynamic allocation is enabled for the cluster, you might see more "
                          "executors allocated.")
        return n_jobs

    def _get_max_num_concurrent_tasks(self):
        # maxNumConcurrentTasks() is a package private API
        # pylint: disable=W0212
        pyspark_version = VersionUtils.majorMinorVersion(pyspark.__version__)
        spark_context = self._spark.sparkContext._jsc.sc()
        if pyspark_version < (3, 1):
            return spark_context.maxNumConcurrentTasks()
        return spark_context.maxNumConcurrentTasks(
                 spark_context.resourceProfileManager().resourceProfileFromId(0))

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
        self.parallel = parallel  # pylint: disable=attribute-defined-outside-init
        return n_jobs

    def _get_pool(self):
        """Lazily initialize the thread pool
        The actual pool of worker threads is only initialized at the first
        call to apply_async.
        """
        if self._pool is None:
            self._pool = ThreadPool(self._n_jobs)
            atexit.register(self._pool.close)
        return self._pool

    def start_call(self):
        self._is_running = True
        if self._ipython is not None:
            def on_post_run_cell(result):
                try:
                    if result.error_in_exec is not None:
                        self._cancel_all_jobs()
                finally:
                    self._ipython.events.unregister("post_run_cell", on_post_run_cell)

            self._ipython.events.register("post_run_cell", on_post_run_cell)

    def apply_async(self, func, callback=None):
        # Note the `func` args is a batch here. (BatchedCalls type)
        # See joblib.parallel.Parallel._dispatch

        def run_on_worker_and_fetch_result():
            if not self._is_running:
                raise RuntimeError('The task is canceled due to ipython command canceled.')

            # TODO: handle possible spark exception here. # pylint: disable=fixme
            worker_rdd = self._spark.sparkContext.parallelize([0], 1)
            if self._resource_profile:
                worker_rdd = worker_rdd.withResources(self._resource_profile)
            def mapper_fn(_):
                return cloudpickle.dumps(func())
            if self._spark_supports_job_cancelling:
                if self._spark_pinned_threads_enabled:
                    self._spark.sparkContext.setLocalProperty(
                        "spark.jobGroup.id",
                        self._job_group
                    )
                    self._spark.sparkContext.setLocalProperty(
                        "spark.job.description",
                        "joblib spark jobs"
                    )
                    rdd = worker_rdd.map(mapper_fn)
                    ser_res = rdd.collect()[0]
                else:
                    rdd = worker_rdd.map(mapper_fn)
                    ser_res = rdd.collectWithJobGroup(
                        self._job_group,
                        "joblib spark jobs"
                    )[0]
            else:
                rdd = worker_rdd.map(mapper_fn)
                ser_res = rdd.collect()[0]

            return cloudpickle.loads(ser_res)

        try:
            # pylint: disable=no-name-in-module,import-outside-toplevel
            from pyspark import inheritable_thread_target
            run_on_worker_and_fetch_result = \
                inheritable_thread_target(run_on_worker_and_fetch_result)
        except ImportError:
            pass

        if SafeFunction is None:
            # pylint: disable=protected-access,used-before-assignment
            return self._get_pool().apply_async(
                PoolManagerMixin._wrap_func_call, (run_on_worker_and_fetch_result,),
                callback=callback, error_callback=callback
            )

        return self._get_pool().apply_async(
            SafeFunction(run_on_worker_and_fetch_result),
            callback=callback, error_callback=callback
        )


    def get_nested_backend(self):
        """Backend instance to be used by nested Parallel calls.
           For nested backend, always use `SequentialBackend`
        """
        nesting_level = getattr(self, 'nesting_level', 0) + 1
        return SequentialBackend(nesting_level=nesting_level), None
