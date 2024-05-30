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
The utils functions for joblib spark backend.
"""
from packaging.version import Version
import pyspark


# pylint: disable=import-outside-toplevel
def get_spark_session():
    """
    Get the spark session from the active session or create a new one.
    """
    from pyspark.sql import SparkSession

    spark_session = SparkSession.getActiveSession()
    if spark_session is None:
        spark_session = SparkSession \
            .builder \
            .appName("JoblibSparkBackend") \
            .getOrCreate()
    return spark_session


def create_resource_profile(num_cpus_per_spark_task, num_gpus_per_spark_task):
    """
    Create a resource profile for the task.
    :param num_cpus_per_spark_task: Number of cpus for each Spark task of current spark job stage.
    :param num_gpus_per_spark_task: Number of gpus for each Spark task of current spark job stage.
    :return: Spark ResourceProfile
    """
    resource_profile = None
    if Version(pyspark.__version__).release >= (3, 4, 0):
        try:
            from pyspark.resource.profile import ResourceProfileBuilder
            from pyspark.resource.requests import TaskResourceRequests
        except ImportError:
            pass
        task_res_req = TaskResourceRequests().cpus(num_cpus_per_spark_task)
        if num_gpus_per_spark_task > 0:
            task_res_req = task_res_req.resource("gpu", num_gpus_per_spark_task)
        resource_profile = ResourceProfileBuilder().require(task_res_req).build
    return resource_profile
