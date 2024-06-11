import warnings
import os
from packaging.version import Version
import unittest
from unittest.mock import MagicMock

import pyspark
from pyspark.sql import SparkSession

from joblibspark.backend import SparkDistributedBackend


class TestLocalSparkCluster(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.spark = (
            SparkSession.builder.master("local[*]").getOrCreate()
        )

    @classmethod
    def teardown_class(cls):
        cls.spark.stop()

    def test_effective_n_jobs(self):
        backend = SparkDistributedBackend()
        max_num_concurrent_tasks = 8
        backend._get_max_num_concurrent_tasks = MagicMock(return_value=max_num_concurrent_tasks)

        assert backend.effective_n_jobs(n_jobs=None) == 1
        assert backend.effective_n_jobs(n_jobs=-1) == 8
        assert backend.effective_n_jobs(n_jobs=4) == 4

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert backend.effective_n_jobs(n_jobs=16) == 16
            assert len(w) == 1

    def test_resource_profile_supported(self):
        backend = SparkDistributedBackend()
        # The test fixture uses a local (standalone) Spark instance, which doesn't support stage-level scheduling.
        assert not backend._support_stage_scheduling


class TestBasicSparkCluster(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.num_cpus_per_spark_task = 1
        cls.num_gpus_per_spark_task = 1
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

    @unittest.skipIf(Version(pyspark.__version__).release < (3, 4, 0),
                     "Resource group is only supported since spark 3.4.0")
    def test_resource_profile(self):
        backend = SparkDistributedBackend(
            num_cpus_per_spark_task=self.num_cpus_per_spark_task,
            num_gpus_per_spark_task=self.num_gpus_per_spark_task)

        assert backend._support_stage_scheduling

        resource_group = backend._resource_profile
        assert resource_group.taskResources['cpus'].amount == 1.0
        assert resource_group.taskResources['gpu'].amount == 1.0

    @unittest.skipIf(Version(pyspark.__version__).release < (3, 4, 0),
                     "Resource group is only supported since spark 3.4.0")
    def test_resource_with_default(self):
        backend = SparkDistributedBackend()

        assert backend._support_stage_scheduling

        resource_group = backend._resource_profile
        assert resource_group.taskResources['cpus'].amount == 1.0
