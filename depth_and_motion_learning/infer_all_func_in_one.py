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

from absl import flags
from absl import app
from absl import logging
import json

from depth_and_motion_learning import depth_motion_field_model
from depth_and_motion_learning.parameter_container import ParameterContainer
#from depth_and_motion_learning import training_utils

import cv2
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'TensorFlow session address.')

flags.DEFINE_string('model_dir', '', 'Directory where the model is saved.')

flags.DEFINE_string('param_overrides', '', 'Parameters for the trainer and the '
                                           'model')

TRAINER_PARAMS = {
    # Learning rate
    'learning_rate': 2e-4,

    # If not None, gradients will be clipped to this value.
    'clip_gradients': 10.0,

    # Number of iterations in the TPU internal on-device loop.
    'iterations_per_loop': 20,

    # If not None, the training will be initialized form this checkpoint.
    'init_ckpt': None,

    # A string, specifies the format of a checkpoint form which to initialize.
    # The model code is expected to convert this string into a
    # vars_to_restore_fn (see below),
    'init_ckpt_type': None,

    # Master address
    'master': None,

    # Directory where checkpoints will be saved.
    'model_dir': None,

    # Maximum number of training steps.
    'max_steps': int(1e6),

    # Number of hours between each checkpoint to be saved.
    # The default value of 10,000 hours effectively disables the feature.
    'keep_checkpoint_every_n_hours': 10000,
}


class InitFromCheckpointHook(tf.estimator.SessionRunHook):
    """A hook for initializing training from a checkpoint.

  Although the Estimator framework supports initialization from a checkpoint via
  https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings,
  the only way to build mapping between the variables and the checkpoint names
  is via providing a regex. This class provides the same functionality, but the
  mapping can be built by a callback, which provides more flexibility and
  readability.
  """

    def __init__(self, model_dir, ckpt_to_init_from, vars_to_restore_fn=None):
        """Creates an instance.

    Args:
      model_dir: A string, path where checkpoints are saved during training.
        Used for checking whether a checkpoint already exists there, in which
        case we want to continue training from there rather than initialize from
        another checkpoint.
      ckpt_to_init_from: A string, path to a checkpoint to initialize from.
      vars_to_restore_fn: A callable that receives no arguments. When called,
        expected to provide a dictionary that maps the checkpoint name of each
        variable to the respective variable object. This dictionary will be used
        as `var_list` in a Saver object used for initializing from
        `ckpt_to_init_from`. If None, the default saver will be used.
    """
        self._ckpt = None if tf.train.latest_checkpoint(
            model_dir) else ckpt_to_init_from
        self._vars_to_restore_fn = vars_to_restore_fn

    def begin(self):
        if not self._ckpt:
            return
        logging.info('%s will be used for initialization.', self._ckpt)
        # Build a saver object for initializing from a checkpoint, or use the
        # default one if no vars_to_restore_fn was given.
        self._reset_step = None
        if tf.train.get_global_step() is not None:
            self._reset_step = tf.train.get_global_step().assign(0)
        if not self._vars_to_restore_fn:
            logging.info('All variables will be initialized form the checkpoint.')
            self._saver = tf.get_collection(tf.GraphKeys.SAVERS)[0]
            return

        vars_to_restore = self._vars_to_restore_fn()
        restored_vars_string = (
            'The following variables are to be initialized from the checkpoint:\n')
        for ckpt_name in sorted(vars_to_restore):
            restored_vars_string += '%s --> %s\n' % (
                ckpt_name, vars_to_restore[ckpt_name].op.name)

        logging.info(restored_vars_string)
        self._saver = tf.train.Saver(vars_to_restore)

    def after_create_session(self, session, coord):
        del coord  # unused
        if not self._ckpt:
            return
        self._saver.restore(session, self._ckpt)
        self._saver.restore(session, self._ckpt)
        if self._reset_step is not None:
            session.run(self._reset_step)


def input_fn_infer(input_image):
  return tf.estimator.inputs.numpy_input_fn(x={"rgb": input_image}, num_epochs=1, shuffle=False)

def serving_input_receiver_fn():
  """For the sake of the example, let's assume your input to the network will be a 28x28 grayscale image that you'll then preprocess as needed"""
  input_images = tf.placeholder(dtype=tf.float32,
                                         shape=[128, 416, 3],
                                         name='input_images')
  # here you do all the operations you need on the images before they can be fed to the net (e.g., normalizing, reshaping, etc). Let's assume "images" is the resulting tensor.

  receiver_tensors = {'input_data': input_images} # As I understand, this will be my access point to the graph

  image = tf.expand_dims(input_images, axis=0)

  features = {'rgb' : image} # this is the dict that is then passed as "features" parameter to your model_fn
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def estimator_spec_fn_infer(features, labels, mode, params):
    del labels #unused

    # depth estimation output of network
    print("IN estimator_spec_fn_infer")
    print(features['rgb'].shape)
    depth_net_out = depth_motion_field_model.infer_depth(rgb_image=features['rgb'], params=params)

    export_outputs = {
      'exported_output': tf.estimator.export.PredictOutput(
        {
          'depth_prediction_output': depth_net_out
        }
      )
    }

    return(tf.estimator.EstimatorSpec(mode=mode, predictions=depth_net_out, export_outputs=export_outputs))


def run_local_inference(input_fn,
                       trainer_params_overrides,
                       model_params,
                       vars_to_restore_fn=None):
    """Run a simple single-mechine traing loop.

  Args:
        input_fn: A callable that complies with tf.Estimtor's definition of
      input_fn.
    trainer_params_overrides: A dictionary or a ParameterContainer with
      overrides for the default values in TRAINER_PARAMS above.
    model_params: A ParameterContainer that will be passed to the model (i. e.
      to losses_fn and input_fn).
    vars_to_restore_fn: A callable that receives no arguments. When called,
      expected to provide a dictionary that maps the checkpoint name of each
      variable to the respective variable object. This dictionary will be used
      as `var_list` in a Saver object used for initializing from the checkpoint
      at trainer_params.init_ckpt. If None, the default saver will be used.
  """
    print("\nIN run_local_inference\n")
    trainer_params = ParameterContainer.from_defaults_and_overrides(
        TRAINER_PARAMS, trainer_params_overrides, is_strict=True)

    run_config_params = {
        'model_dir':
            trainer_params.model_dir,
        'save_summary_steps':
            5,
        'keep_checkpoint_every_n_hours':
            trainer_params.keep_checkpoint_every_n_hours,
        'log_step_count_steps':
            25,
    }
    logging.info(
        'Estimators run config parameters:\n%s',
        json.dumps(run_config_params, indent=2, sort_keys=True, default=str))
    run_config = tf.estimator.RunConfig(**run_config_params)


    init_hook = InitFromCheckpointHook(trainer_params.model_dir,
                                       trainer_params.init_ckpt,
                                       vars_to_restore_fn)

    estimator = tf.estimator.Estimator(
        model_fn=estimator_spec_fn_infer,
        config=run_config,
        params=model_params.as_dict())

    print("\n\nPRE estimator.predict")
    predict_output = estimator.predict(input_fn=input_fn, predict_keys=None, hooks=[init_hook])
    predict_output_np = np.array(list(predict_output))
    
    hist, bins = np.histogram(predict_output_np, bins=range(0,255))
    import matplotlib.pyplot as plt
    plt.plot(bins[:-1], hist)
    plt.savefig("/home/fascar/Documents/mono_depth/data/hist.png")


    depth_img = np.reshape(predict_output_np, (128, 416)) 
    max_val = np.max(depth_img)
    depth_img = depth_img * (255/max_val)
    cv2.imwrite("/home/fascar/Documents/mono_depth/data/2020-10-22-10-29-48_7_36_output.png", depth_img)
    #cv2.imshow("depth", depth_img)
    
    print("\n\nPOST estimator.predict")

    print("\nOUT run_local_inference")

    print("\n\nPRE EXPORT MODEL")
    estimator.export_savedmodel("/home/fascar/Documents/mono_depth/models", serving_input_receiver_fn=serving_input_receiver_fn)
    print("\n\nPOST EXPORT MODEL")


def infer(input_fn, get_vars_to_restore_fn=None):
    """Run inference.

  Args:
    input_fn: A tf.Estimator compliant input_fn.
        get_vars_to_restore_fn: A callable that receives a string argument
      (intdicating the type of initialization) and returns a vars_to_restore_fn.
      The latter is a callable that receives no arguments and returns a
      dictionary that can be passed to a tf.train.Saver object's constructor as
      a `var_list` to indicate which variables to load from what names in the
      checnpoint.
  """
    params = ParameterContainer({
        'model': {
            'batch_size': 1,
            'input': {}
        },
    }, {'trainer': {
        'master': FLAGS.master,
        'model_dir': FLAGS.model_dir
    }})

    params.override(FLAGS.param_overrides)

    init_ckpt_type = params.trainer.get('init_ckpt_type')

    if init_ckpt_type and not get_vars_to_restore_fn:
        raise ValueError(
            'An init_ckpt_type was specified (%s), but no get_vars_to_restore_fn '
            'was provided.' % init_ckpt_type)

    vars_to_restore_fn = (
        get_vars_to_restore_fn(init_ckpt_type) if init_ckpt_type else None)

    logging.info(
        'Starting training with the following parameters:\n%s',
        json.dumps(params.as_dict(), indent=2, sort_keys=True, default=str))

    run_local_inference(input_fn, params.trainer, params.model,
                       vars_to_restore_fn)





def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  image_file = "/home/fascar/Documents/mono_depth/data/2020-10-22-10-29-48_7_36.png"
  filename = "/home/fascar/Documents/mono_depth/data/2020-10-22-10-29-48_7_36"
  input_image = cv2.imread(image_file).astype(np.float32)
  input_image = input_image #* (1 / 255.0)


  np.save(filename + ".npy" , np.array(input_image, dtype='float32'))

  encoded_image = tf.io.read_file(image_file)
  decoded_image = tf.image.decode_png(encoded_image, channels=3)
  decoded_image = tf.to_float(decoded_image) * (1 / 255.0)

  input_batch = np.reshape(input_image, (1, 128, 416, 3))

  infer(input_fn_infer(input_image=input_batch),
                       depth_motion_field_model.get_vars_to_restore_fn)


if __name__ == '__main__':
  app.run(main)

