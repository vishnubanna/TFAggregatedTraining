"""Normalization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.distribute import reduce_util
from tensorflow.python.keras.layers import normalization



import tensorflow as tf


class ShuffleBatchNormalization(normalization.BatchNormalizationBase):
  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               trainable=True,
               adjustment=None,
               name=None,
               **kwargs):

    # Currently we only support aggregating over the global batch size.
    super(ShuffleBatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=False,
        trainable=trainable,
        virtual_batch_size=None, 
        name=name,
        **kwargs)
  def _calculate_mean_and_var(self, x, axes, keep_dims):
  
    with K.name_scope('moments'):
      # The dynamic range of fp16 is too limited to support the collection of
      # sufficient statistics. As a workaround we simply perform the operations
      # on 32-bit floats before converting the mean and variance back to fp16
      y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
      # if you only have one replica dont worry about it 
      # Compute true mean while keeping the dims for proper broadcasting.
      mean = math_ops.reduce_mean(y, axes, keepdims=True, name='mean')
      # sample variance, not unbiased variance
      # Note: stop_gradient does not change the gradient that gets
      #       backpropagated to the mean from the variance calculation,
      #       because that gradient is zero
      variance = math_ops.reduce_mean(
          math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
          axes,
          keepdims=True,
          name='variance')

      replica_ctx = ds.get_replica_context()
      if replica_ctx:
        tf.print(replica_ctx.num_replicas_in_sync)
        tf.print(replica_ctx.replica_id_in_sync_group)
  

      if not keep_dims:
        mean = array_ops.squeeze(mean, axes)
        variance = array_ops.squeeze(variance, axes)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(variance, dtypes.float16))
      else:
        return (mean, variance)