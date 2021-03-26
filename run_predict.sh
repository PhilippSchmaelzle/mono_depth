# Copyright 2021 The Google Research Authors.
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

#!/bin/bash
set -e
set -x

#virtualenv -p python3 .
#source ./bin/activate

#pip install tensorflow==1.15.0
#pip install tensorflow-graphics==1.0.0
#pip install matplotlib==3.3.0
#pip install -r depth_and_motion_learning/requirements.txt


#"data_path": "depth_from_video_in_the_wild/data_example/train.txt"

python3.7 -m depth_and_motion_learning.infer_all_func_in_one \
  --model_dir=/home/fascar/Documents/mono_depth/models/cityscape_plus_avt \
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "/home/fascar/Documents/mono_depth/data/test.txt"
      }
    },
    "trainer": {
      "init_ckpt": "/home/fascar/Documents/mono_depth_training/models/resnet18_ckpt_from_torch/model.ckpt",
      "init_ckpt_type": "imagenet",
      "max_steps": 125001
    }
  }'
