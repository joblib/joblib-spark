import warnings
from unittest.mock import MagicMock

from joblibspark.backend import SparkDistributedBackend


def test_effective_n_jobs():

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
