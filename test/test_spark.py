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
from sklearn.externals.joblib.parallel \
    import Parallel, delayed, parallel_backend
from time import sleep
import pytest


def inc(x):
    return x + 1


def slow_raise_value_error(condition, duration=0.05):
    sleep(duration)
    if condition:
        raise ValueError("condition evaluated to True")


def test_simple():
    with parallel_backend('dask') as (ba, _):
        seq = Parallel(n_jobs=5)(delayed(inc)(i) for i in range(10))
        assert seq == [inc(i) for i in range(10)]

    with pytest.raises(BaseException):
        Parallel(n_jobs=5)(delayed(slow_raise_value_error)(i == 3)
                           for i in range(10))
