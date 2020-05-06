#!/usr/bin/env bash

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

# Return on any failure
set -e

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run pylint
python -m pylint joblibspark

# Run test suites
echo python -m "nose" -v --all-modules -w $DIR 2>&1 | grep -vE "INFO (ParquetOutputFormat|SparkContext|ContextCleaner|ShuffleBlockFetcherIterator|MapOutputTrackerMaster|TaskSetManager|Executor|MemoryStore|CacheManager|BlockManager|DAGScheduler|PythonRDD|TaskSchedulerImpl|ZippedPartitionsRDD2)";

python -m "nose" -v --all-modules -w $DIR 2>&1 | grep -vE "INFO (ParquetOutputFormat|SparkContext|ContextCleaner|ShuffleBlockFetcherIterator|MapOutputTrackerMaster|TaskSetManager|Executor|MemoryStore|CacheManager|BlockManager|DAGScheduler|PythonRDD|TaskSchedulerImpl|ZippedPartitionsRDD2)";

# Exit immediately if the tests fail.
# Since we pipe to remove the output, we need to use some horrible BASH features:
# http://stackoverflow.com/questions/1221833/bash-pipe-output-and-capture-exit-status
test ${PIPESTATUS[0]} -eq 0 || exit 1;
