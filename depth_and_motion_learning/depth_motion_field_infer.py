# coding=utf-8
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

"""A binary for training depth and egomotion."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app

from depth_and_motion_learning import depth_motion_field_model
from depth_and_motion_learning import training_utils

import cv2
import numpy as np
import tensorflow as tf


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  image_file = "/home/fascar/Documents/mono_depth/data/2020-10-22-10-29-48_7_36.png"
  input_image = cv2.imread(image_file).astype(np.float32)
  input_image = input_image #* (1 / 255.0)

  encoded_image = tf.io.read_file(image_file)
  decoded_image = tf.image.decode_png(encoded_image, channels=3)
  decoded_image = tf.to_float(decoded_image) * (1 / 255.0)

  print("loooooooooooooooooooooooooooool")
  print(input_image)
  input_batch = np.reshape(input_image, (1, 128, 416, 3))
  cv2.imshow("input", np.reshape(input_batch, (128,416,3)))
  cv2.waitKey(3)

  training_utils.infer(depth_motion_field_model.input_fn_infer(input_image=input_batch),
                       depth_motion_field_model.loss_fn,
                       depth_motion_field_model.get_vars_to_restore_fn)


if __name__ == '__main__':
  app.run(main)
