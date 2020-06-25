import contextlib
import logging
from unittest.mock import MagicMock
from six import StringIO

from joblibspark.backend import SparkDistributedBackend


@contextlib.contextmanager
def patch_logger(name, level=logging.INFO):
    """patch logger and give an output"""
    io_out = StringIO()
    log = logging.getLogger(name)
    log.setLevel(level)
    log.handlers = []
    handler = logging.StreamHandler(io_out)
    log.addHandler(handler)
    try:
        yield io_out
    finally:
        log.removeHandler(handler)


def test_effective_n_jobs():

    backend = SparkDistributedBackend()
    max_num_concurrent_tasks = 8
    backend._get_max_num_concurrent_tasks = MagicMock(return_value=max_num_concurrent_tasks)

    assert backend.effective_n_jobs(n_jobs=None) == 1
    assert backend.effective_n_jobs(n_jobs=-1) == 8
    assert backend.effective_n_jobs(n_jobs=4) == 4
    assert backend.effective_n_jobs(n_jobs=16) == 16


def test_parallelism_arg():
    for spark_default_parallelism, max_num_concurrent_tasks in [(2, 4), (2, 0)]:
        default_parallelism = max(spark_default_parallelism, max_num_concurrent_tasks)

        assert 1 == SparkDistributedBackend._decide_parallelism(
            requested_parallelism=None,
            spark_default_parallelism=spark_default_parallelism,
            max_num_concurrent_tasks=max_num_concurrent_tasks,
        )
        with patch_logger("joblib-spark") as output:
            parallelism = SparkDistributedBackend._decide_parallelism(
                requested_parallelism=-1,
                spark_default_parallelism=spark_default_parallelism,
                max_num_concurrent_tasks=max_num_concurrent_tasks,
            )
            assert parallelism == default_parallelism
            log_output = output.getvalue().strip()
            assert "Because the requested parallelism was None or a non-positive value, " \
                "parallelism will be set to ({d})".format(d=default_parallelism) in log_output

        # Test requested_parallelism which will trigger spark executor dynamic allocation.
        with patch_logger("joblib-spark") as output:
            parallelism = SparkDistributedBackend._decide_parallelism(
                requested_parallelism=max_num_concurrent_tasks + 1,
                spark_default_parallelism=spark_default_parallelism,
                max_num_concurrent_tasks=max_num_concurrent_tasks,
            )
            assert parallelism == max_num_concurrent_tasks + 1
            log_output = output.getvalue().strip()
            assert "Parallelism ({p}) is greater".format(p=max_num_concurrent_tasks + 1) \
                   in log_output

        # Test requested_parallelism exceeds hard cap
        with patch_logger("joblib-spark") as output:
            parallelism = SparkDistributedBackend._decide_parallelism(
                requested_parallelism=SparkDistributedBackend.MAX_CONCURRENT_JOBS_ALLOWED + 1,
                spark_default_parallelism=spark_default_parallelism,
                max_num_concurrent_tasks=max_num_concurrent_tasks,
            )
            assert parallelism == SparkDistributedBackend.MAX_CONCURRENT_JOBS_ALLOWED
            log_output = output.getvalue().strip()
            assert "SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED ({c})" \
                .format(c=SparkDistributedBackend.MAX_CONCURRENT_JOBS_ALLOWED) in log_output
