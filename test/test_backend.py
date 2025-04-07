import warnings
import os
from packaging.version import Version
import unittest
from unittest.mock import MagicMock

import pyspark
from pyspark.sql import SparkSession

from joblibspark.backend import SparkDistributedBackend
import joblibspark.backend

joblibspark.backend._DEFAULT_N_JOBS_IN_SPARK_CONNECT_MODE = 8


spark_version = os.environ["PYSPARK_VERSION"]
is_spark_connect_mode = os.environ["TEST_SPARK_CONNECT"].lower() == "true"


if Version(spark_version).major >= 4:
    spark_connect_jar = ""
else:
    spark_connect_jar = f"org.apache.spark:spark-connect_2.12:{spark_version}"


class TestLocalSparkCluster(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        if is_spark_connect_mode:
            cls.spark = (
                SparkSession.builder.config(
                    "spark.jars.packages", spark_connect_jar
                )
                    .remote("local-cluster[1, 2, 1024]")
                    .appName("Test")
                    .getOrCreate()
            )
        else:
            cls.spark = (
                SparkSession.builder.master("local-cluster[1, 2, 1024]").getOrCreate()
            )

    @classmethod
    def teardown_class(cls):
        cls.spark.stop()

    def test_effective_n_jobs(self):
        backend = SparkDistributedBackend()

        assert backend.effective_n_jobs(n_jobs=None) == 1
        assert backend.effective_n_jobs(n_jobs=4) == 4

        if is_spark_connect_mode:
            assert (
                backend.effective_n_jobs(n_jobs=-1) ==
                joblibspark.backend._DEFAULT_N_JOBS_IN_SPARK_CONNECT_MODE
            )
        else:
            max_num_concurrent_tasks = 8
            backend._get_max_num_concurrent_tasks = MagicMock(return_value=max_num_concurrent_tasks)
            assert backend.effective_n_jobs(n_jobs=-1) == 8
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                assert backend.effective_n_jobs(n_jobs=16) == 16
                assert len(w) == 1


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
