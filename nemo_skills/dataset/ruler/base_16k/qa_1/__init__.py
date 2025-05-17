# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# settings that define how evaluation should be done by default (all can be changed from cmdline)
# this will be copied and updated for each ruler subtask. 
# PROMPT_CONFIG = 'generic/default_ruler'
# DATASET_GROUP = "chat"
# METRICS_TYPE = "ruler"
# DEFAULT_EVAL_ARGS = "++eval_type=ruler ++eval_config.match_type=all"
# DEFAULT_GENERATION_ARGS = "++inference.tokens_to_generate=50 ++input_file=/ruler/ruler_llama_4k/fwe/test.jsonl ++dataset=null ++split=null"
PROMPT_CONFIG = "generic/default_ruler"
DATASET_GROUP = "chat"
METRICS_TYPE = "ruler"
DEFAULT_EVAL_ARGS = "++eval_type=ruler ++eval_config.match_type=part"
DEFAULT_GENERATION_ARGS = "++inference.tokens_to_generate=32 ++input_file=/ruler/base_16k/qa_1/test.jsonl ++dataset=null ++split=null"
