import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import Model

# Custom advection block 1 - for input velocity vector fields stored at cell boundaries (c/d-grid)
class AdvecBlock1(tf.keras.Model):
  def __init__(self, filters, square=True, mult_wind=False, stack_wind=False, padding_offset=0, extra_conv=False, smooth_padd=False, flip_indicators=True, num_derivs=1):
    super(AdvecBlock1, self).__init__()
    self.filters_1 = filters
    self.square = square
    self.mult_wind = mult_wind
    self.stack_wind = stack_wind
    self.padding_offset = padding_offset
    self.extra_conv = extra_conv
    self.smooth_padd = smooth_padd
    self.flip_indicators = flip_indicators
    self.num_derivs = num_derivs

    # create padding for our convolutional layers
    self.padd_ul = ZeroPadding2D(padding=((1, 1), (1 - self.padding_offset, 1 + padding_offset)))
    self.padd_ur = ZeroPadding2D(padding=((1, 1), (1 + self.padding_offset, 1 - padding_offset)))
    self.padd_vl = ZeroPadding2D(padding=((1 - self.padding_offset, 1 + self.padding_offset), (1, 1)))
    self.padd_vr = ZeroPadding2D(padding=((1 + self.padding_offset, 1 - self.padding_offset), (1, 1)))
    # example u_l for padding_offset==-1:
    # 0 0 0 0
    # 0 0 X X
    # 0 0 X X
    # 0 0 0 0

    # if smooth_padd == False then we will use centered padding when creating smoothness indicators. we also use it for reducing dimensionality
    self.padd_center = ZeroPadding2D(padding=((1, 1), (1, 1)))

    # we will now fill a few lists containing the convolutions we will use (with separate weights)
    self.ul_convolutions = []
    self.ur_convolutions = []
    self.vl_convolutions = []
    self.vr_convolutions = []
    for i in range(self.num_derivs + 3 + self.extra_conv):
      # we are including 3 extra convolutions: one to apply to concentration, one for the appropriate wind vector component, and one for the product
      # we also apply an additional convolutional layer if extra_conv == True
      self.ul_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
      self.ur_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
      self.vl_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
      self.vr_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
    self.conv_indicators = Conv2D(self.filters_1 * 4, (3, 3), padding='valid')
    self.conv_out = Conv2D(self.filters_1, (1, 1), padding='valid')

    # if flip_indicators == True we will invert our smoothness indicators before feeding it into the softmax layer
    if self.flip_indicators == True:
      self.flip = Lambda(lambda b: 1/(b + K.epsilon()))

    self.mult = Multiply()

    # if square == True we will square our smoothness indicators

    if self.square == True:
      self.square = Lambda(lambda b: K.square(b))
    self.soft = Softmax(axis=-1)
  def call(self, input_tensor, training=False):
    p = input_tensor[0]
    ul, ur, vl, vr = [tf.slice(input_tensor[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(4)]

    # first we deal with the left boundary of the u velocity component

    # establish which padding we will be using
    padd_ul = self.padd_ul
    # apply a convolution to the concentration field
    p_ul = self.ul_convolutions[-1](padd_ul(p))
    # apply a separate convolution to the left boundary of the u component of velocity
    ul_1 = self.ul_convolutions[-2](padd_ul(ul))
    # multiply the concentration and appropriate velocity component to get terms estimating mass flux
    mult_ul_1 = self.mult([p_ul, ul_1])
    # apply at least one convolution to this to get various weighted averages of local weighted fluxes
    x_ul_1 = self.ul_convolutions[-3](padd_ul(mult_ul_1))
    a_ul_1 = tf.nn.tanh(x_ul_1)
    # if extra_conv is true apply an extra convolution for larger stencil / more complex relationships among local fluxes
    if self.extra_conv == True:
      x_ul_1 = self.ul_convolutions[-4](padd_ul(mult_ul_1))
      # a_ul_1 will contain the fluxes corresponding to the left boundaries that we will weight later in the network
      a_ul_1 = tf.nn.tanh(x_ul_1)
    # we will now try to find weights for the a_ul_1 feature maps that correspond to smoothness
    indicators_ul = a_ul_1
    # if smooth_padd is false we will switch back to centered padding for the generation of smoothness indicators and weights
    if self.smooth_padd == False:
      padd_ul = self.padd_center
    # now we will estimate spatial derivatives at various orders to form the smoothness indicators
    for i in range(self.num_derivs):
      if self.mult_wind:
        # if mult_wind is true we multiply the spatial derivatives by the proper velocity component
        if i == 0:
          # if i=0 we need to make sure to use the convolutional layer from before this for-loop
          mult_ul = self.mult([a_ul_1, ul_1])
        else:
          mult_ul = self.mult([a_ul, ul_1])
        x_ul = self.ul_convolutions[i](padd_ul(mult_ul))
      else:
        if i == 0:
          x_ul = self.ul_convolutions[i](padd_ul(a_ul_1))
        else:
          x_ul = self.ul_convolutions[i](padd_ul(a_ul))
      # regardless of mult_wind and smooth_padd we will use a tanh layer here
      a_ul = tf.nn.tanh(x_ul)
      # collect this quantity (spatial derivative) at each order (i+1) to be later used for smoothness indicators
      indicators_ul = concatenate([indicators_ul, a_ul], axis=-1)

    # Now we move onto the right boundary of the u velocity component

    padd_ur = self.padd_ur
    p_ur = self.ur_convolutions[-1](padd_ur(p))
    ur_1 = self.ur_convolutions[-2](padd_ur(ur))
    mult_ur_1 = self.mult([p_ur, ur_1])
    x_ur_1 = self.ur_convolutions[-3](padd_ur(mult_ur_1))
    a_ur_1 = tf.nn.tanh(x_ur_1)
    if self.extra_conv == True:
      x_ur_1 = self.ur_convolutions[-4](padd_ur(mult_ur_1))
      a_ur_1 = tf.nn.tanh(x_ur_1)
    indicators_ur = a_ur_1
    if self.smooth_padd == False:
      # use center padding for derivative estimates used for smoothness indicators, not offset paddings
      padd_ur = self.padd_center
    for i in range(self.num_derivs):
      if self.mult_wind:
        if i == 0:
          mult_ur = self.mult([a_ur_1, ur_1])
        else:
          mult_ur = self.mult([a_ur, ur_1])
        x_ur = self.ur_convolutions[i](padd_ur(mult_ur))
      else:
        if i == 0:
          x_ur = self.ur_convolutions[i](padd_ur(a_ur_1))
        else:
          x_ur = self.ur_convolutions[i](padd_ur(a_ur))
      a_ur = tf.nn.tanh(x_ur)
      indicators_ur = concatenate([indicators_ur, a_ur], axis=-1)

    # Now we move onto the top boundary of the v velocity component

    padd_vl = self.padd_vl
    p_vl = self.vl_convolutions[-1](padd_vl(p))
    vl_1 = self.vl_convolutions[-2](padd_vl(vl))
    mult_vl_1 = self.mult([p_vl, vl_1])
    x_vl_1 = self.vl_convolutions[-3](padd_vl(mult_vl_1))
    a_vl_1 = tf.nn.tanh(x_vl_1)
    if self.extra_conv == True:
      x_vl_1 = self.vl_convolutions[-4](padd_vl(mult_vl_1))
      a_vl_1 = tf.nn.tanh(x_vl_1)
    indicators_vl = a_vl_1
    if self.smooth_padd == False:
      # use center padding for derivative estimates used for smoothness indicators, not offset paddings
      padd_vl = self.padd_center
    for i in range(self.num_derivs):
      if self.mult_wind:
        # if mult_wind is true we multiply the smoothness indicators by the proper velocity component
        if i == 0:
          # if i=0 we need to make sure to use the convolutional layer from before this for loop
          mult_vl = self.mult([a_vl_1, vl_1])
        else:
          mult_vl = self.mult([a_vl, vl_1])
        x_vl = self.vl_convolutions[i](padd_vl(mult_vl))
      else:
        if i == 0:
          x_vl = self.vl_convolutions[i](padd_vl(a_vl_1))
        else:
          x_vl = self.vl_convolutions[i](padd_vl(a_vl))
      a_vl = tf.nn.tanh(x_vl)
      indicators_vl = concatenate([indicators_vl, a_vl], axis=-1)

    # Now we will deal with the bottom boundary of the v velocity component

    padd_vr = self.padd_vr
    p_vr = self.vr_convolutions[-1](padd_vr(p))
    vr_1 = self.vr_convolutions[-2](padd_vr(vr))
    mult_vr_1 = self.mult([p_vr, vr_1])
    x_vr_1 = self.vr_convolutions[-3](padd_vr(mult_vr_1))
    a_vr_1 = tf.nn.tanh(x_vr_1)
    if self.extra_conv == True:
      x_vr_1 = self.vr_convolutions[-4](padd_vr(mult_vr_1))
      a_vr_1 = tf.nn.tanh(x_vr_1)
    indicators_vr = a_vr_1
    if self.smooth_padd == False:
      # use center padding for derivative estimates used for smoothness indicators, not offset paddings
      padd_vr = self.padd_center
    for i in range(self.num_derivs):
      if self.mult_wind:
        # if mult_wind is true we multiply the smoothness indicators by the proper velocity component
        if i == 0:
          # if i=0 we need to make sure to use the convolutional layer from before this for loop
          mult_vr = self.mult([a_vr_1, vr_1])
        else:
          mult_vr = self.mult([a_vr, vr_1])
        x_vr = self.vr_convolutions[i](padd_vr(mult_vr))
      else:
        if i == 0:
          x_vr = self.vr_convolutions[i](padd_vr(a_vr_1))
        else:
          x_vr = self.vr_convolutions[i](padd_vr(a_vr))
      a_vr = tf.nn.tanh(x_vr)
      indicators_vr = concatenate([indicators_vr, a_vr], axis=-1)

    concat = concatenate([indicators_ul, indicators_ur, indicators_vl, indicators_vr], axis=-1)
    # if square == True we will square the feature maps used for the smoothness indicators
    if self.square:
      concat = self.square(concat)
    # reduce the dimensionality so we have weights for each feature map of each velocity component
    indicators = self.conv_indicators(self.padd_center(concat))
    # if flip == True we will flip the indicators to get weights
    if self.flip:
      indicators = self.flip(indicators)
    # we use a softmax layer to get normalized relative weights for the various feature maps of each velocity component
    weights = self.soft(indicators)
    # now we will collect the feature maps that we will weight
    concat_2 = concatenate([a_ul_1, a_ur_1, a_vl_1, a_vr_1], axis=-1)
    # weight the feature maps with a multiplication layer
    mult = self.mult([weights, concat_2])
    # if stack_wind == True we will include the velocity components as standalone features when forming the output
    if self.stack_wind == True:
      mult = concatenate([concat_2, ul, ur, vl, vr])
    out = self.conv_out(mult)
    return tf.nn.tanh(out)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# second block for data with cell-centered estimates for velocity components

class AdvecBlock2(tf.keras.Model):
  def __init__(self, filters, square=True, mult_wind=False, stack_wind=False, padding_offset=0, extra_conv=False, smooth_padd=False, flip_indicators=True, num_derivs=1):
    super(AdvecBlock2, self).__init__()
    self.filters_1 = filters
    self.square = square
    self.mult_wind = mult_wind
    self.stack_wind = stack_wind
    self.padding_offset = padding_offset
    self.extra_conv = extra_conv
    self.smooth_padd = smooth_padd
    self.flip_indicators = flip_indicators
    self.num_derivs = num_derivs

    # create padding for our convolutional layers
    self.padd_ul = ZeroPadding2D(padding=((1, 1), (1 - self.padding_offset, 1 + padding_offset)))
    self.padd_ur = ZeroPadding2D(padding=((1, 1), (1 + self.padding_offset, 1 - padding_offset)))
    self.padd_vl = ZeroPadding2D(padding=((1 - self.padding_offset, 1 + self.padding_offset), (1, 1)))
    self.padd_vr = ZeroPadding2D(padding=((1 + self.padding_offset, 1 - self.padding_offset), (1, 1)))
    # example u_l for padding_offset==-1:
    # 0 0 0 0
    # 0 0 X X
    # 0 0 X X
    # 0 0 0 0

    # if smooth_padd == False then we will use centered padding when creating smoothness indicators. we also use it for reducing dimensionality
    self.padd_center = ZeroPadding2D(padding=((1, 1), (1, 1)))

    # we will now fill a few lists containing the convolutions we will use (with separate weights)
    self.ul_convolutions = []
    self.ur_convolutions = []
    self.vl_convolutions = []
    self.vr_convolutions = []
    for i in range(self.num_derivs + 3 + self.extra_conv):
      # we are including 3 extra convolutions: one to apply to concentration, one for the appropriate wind vector component, and one for the product
      # we also apply an additional convolutional layer if extra_conv == True
      self.ul_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
      self.ur_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
      self.vl_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
      self.vr_convolutions.append(Conv2D(self.filters_1, (3, 3), padding='valid'))
    self.conv_indicators = Conv2D(self.filters_1 * 4, (3, 3), padding='valid')
    self.conv_out = Conv2D(self.filters_1, (1, 1), padding='valid')

    # if flip_indicators == True we will invert our smoothness indicators before feeding it into the softmax layer
    if self.flip_indicators == True:
      self.flip = Lambda(lambda b: 1/(b + K.epsilon()))

    self.mult = Multiply()

    # if square == True we will square our smoothness indicators

    if self.square == True:
      self.square = Lambda(lambda b: K.square(b))
    self.soft = Softmax(axis=-1)
  def call(self, input_tensor, training=False):
    p = input_tensor[0]
    u, v = [tf.slice(input_tensor[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)]
    # first we deal with the left boundary of the u velocity component

    # establish which padding we will be using
    padd_ul = self.padd_ul
    # apply a convolution to the concentration field
    p_ul = self.ul_convolutions[-1](padd_ul(p))
    # apply a separate convolution to the left boundary of the u component of velocity
    ul_1 = self.ul_convolutions[-2](padd_ul(u))
    # multiply the concentration and appropriate velocity component to get terms estimating mass flux
    mult_ul_1 = self.mult([p_ul, ul_1])
    # apply at least one convolution to this to get various weighted averages of local weighted fluxes
    x_ul_1 = self.ul_convolutions[-3](padd_ul(mult_ul_1))
    a_ul_1 = tf.nn.tanh(x_ul_1)
    # if extra_conv is true apply an extra convolution for larger stencil / more complex relationships among local fluxes
    if self.extra_conv == True:
      x_ul_1 = self.ul_convolutions[-4](padd_ul(mult_ul_1))
      # a_ul_1 will contain the fluxes corresponding to the left boundaries that we will weight later in the network
      a_ul_1 = tf.nn.tanh(x_ul_1)
    # we will now try to find weights for the a_ul_1 feature maps that correspond to smoothness
    indicators_ul = a_ul_1
    # if smooth_padd is false we will switch back to centered padding for the generation of smoothness indicators and weights
    if self.smooth_padd == False:
      padd_ul = self.padd_center
    # now we will estimate spatial derivatives at various orders to form the smoothness indicators
    for i in range(self.num_derivs):
      if self.mult_wind:
        # if mult_wind is true we multiply the spatial derivatives by the proper velocity component
        if i == 0:
          # if i=0 we need to make sure to use the convolutional layer from before this for-loop
          mult_ul = self.mult([a_ul_1, ul_1])
        else:
          mult_ul = self.mult([a_ul, ul_1])
        x_ul = self.ul_convolutions[i](padd_ul(mult_ul))
      else:
        if i == 0:
          x_ul = self.ul_convolutions[i](padd_ul(a_ul_1))
        else:
          x_ul = self.ul_convolutions[i](padd_ul(a_ul))
      # regardless of mult_wind and smooth_padd we will use a tanh layer here
      a_ul = tf.nn.tanh(x_ul)
      # collect this quantity (spatial derivative) at each order (i+1) to be later used for smoothness indicators
      indicators_ul = concatenate([indicators_ul, a_ul], axis=-1)

    # Now we move onto the right boundary of the u velocity component

    padd_ur = self.padd_ur
    p_ur = self.ur_convolutions[-1](padd_ur(p))
    ur_1 = self.ur_convolutions[-2](padd_ur(u))
    mult_ur_1 = self.mult([p_ur, ur_1])
    x_ur_1 = self.ur_convolutions[-3](padd_ur(mult_ur_1))
    a_ur_1 = tf.nn.tanh(x_ur_1)
    if self.extra_conv == True:
      x_ur_1 = self.ur_convolutions[-4](padd_ur(mult_ur_1))
      a_ur_1 = tf.nn.tanh(x_ur_1)
    indicators_ur = a_ur_1
    if self.smooth_padd == False:
      # use center padding for derivative estimates used for smoothness indicators, not offset paddings
      padd_ur = self.padd_center
    for i in range(self.num_derivs):
      if self.mult_wind:
        if i == 0:
          mult_ur = self.mult([a_ur_1, ur_1])
        else:
          mult_ur = self.mult([a_ur, ur_1])
        x_ur = self.ur_convolutions[i](padd_ur(mult_ur))
      else:
        if i == 0:
          x_ur = self.ur_convolutions[i](padd_ur(a_ur_1))
        else:
          x_ur = self.ur_convolutions[i](padd_ur(a_ur))
      a_ur = tf.nn.tanh(x_ur)
      indicators_ur = concatenate([indicators_ur, a_ur], axis=-1)

    # Now we move onto the top boundary of the v velocity component

    padd_vl = self.padd_vl
    p_vl = self.vl_convolutions[-1](padd_vl(p))
    vl_1 = self.vl_convolutions[-2](padd_vl(v))
    mult_vl_1 = self.mult([p_vl, vl_1])
    x_vl_1 = self.vl_convolutions[-3](padd_vl(mult_vl_1))
    a_vl_1 = tf.nn.tanh(x_vl_1)
    if self.extra_conv == True:
      x_vl_1 = self.vl_convolutions[-4](padd_vl(mult_vl_1))
      a_vl_1 = tf.nn.tanh(x_vl_1)
    indicators_vl = a_vl_1
    if self.smooth_padd == False:
      # use center padding for derivative estimates used for smoothness indicators, not offset paddings
      padd_vl = self.padd_center
    for i in range(self.num_derivs):
      if self.mult_wind:
        # if mult_wind is true we multiply the smoothness indicators by the proper velocity component
        if i == 0:
          # if i=0 we need to make sure to use the convolutional layer from before this for loop
          mult_vl = self.mult([a_vl_1, vl_1])
        else:
          mult_vl = self.mult([a_vl, vl_1])
        x_vl = self.vl_convolutions[i](padd_vl(mult_vl))
      else:
        if i == 0:
          x_vl = self.vl_convolutions[i](padd_vl(a_vl_1))
        else:
          x_vl = self.vl_convolutions[i](padd_vl(a_vl))
      a_vl = tf.nn.tanh(x_vl)
      indicators_vl = concatenate([indicators_vl, a_vl], axis=-1)

    # Now we will deal with the bottom boundary of the v velocity component

    padd_vr = self.padd_vr
    p_vr = self.vr_convolutions[-1](padd_vr(p))
    vr_1 = self.vr_convolutions[-2](padd_vr(v))
    mult_vr_1 = self.mult([p_vr, vr_1])
    x_vr_1 = self.vr_convolutions[-3](padd_vr(mult_vr_1))
    a_vr_1 = tf.nn.tanh(x_vr_1)
    if self.extra_conv == True:
      x_vr_1 = self.vr_convolutions[-4](padd_vr(mult_vr_1))
      a_vr_1 = tf.nn.tanh(x_vr_1)
    indicators_vr = a_vr_1
    if self.smooth_padd == False:
      # use center padding for derivative estimates used for smoothness indicators, not offset paddings
      padd_vr = self.padd_center
    for i in range(self.num_derivs):
      if self.mult_wind:
        # if mult_wind is true we multiply the smoothness indicators by the proper velocity component
        if i == 0:
          # if i=0 we need to make sure to use the convolutional layer from before this for loop
          mult_vr = self.mult([a_vr_1, vr_1])
        else:
          mult_vr = self.mult([a_vr, vr_1])
        x_vr = self.vr_convolutions[i](padd_vr(mult_vr))
      else:
        if i == 0:
          x_vr = self.vr_convolutions[i](padd_vr(a_vr_1))
        else:
          x_vr = self.vr_convolutions[i](padd_vr(a_vr))
      a_vr = tf.nn.tanh(x_vr)
      indicators_vr = concatenate([indicators_vr, a_vr], axis=-1)

    concat = concatenate([indicators_ul, indicators_ur, indicators_vl, indicators_vr], axis=-1)
    # if square == True we will square the feature maps used for the smoothness indicators
    if self.square:
      concat = self.square(concat)
    # reduce the dimensionality so we have weights for each feature map of each velocity component
    indicators = self.conv_indicators(self.padd_center(concat))
    # if flip == True we will flip the indicators to get weights
    if self.flip:
      indicators = self.flip(indicators)
    # we use a softmax layer to get normalized relative weights for the various feature maps of each velocity component
    weights = self.soft(indicators)
    # now we will collect the feature maps that we will weight
    concat_2 = concatenate([a_ul_1, a_ur_1, a_vl_1, a_vr_1], axis=-1)
    # weight the feature maps with a multiplication layer
    mult = self.mult([weights, concat_2])
    # if stack_wind == True we will include the velocity components as standalone features when forming the output
    if self.stack_wind == True:
      mult = concatenate([concat_2, u, v])
    out = self.conv_out(mult)
    return tf.nn.tanh(out)

class AdvecBlock3(tf.keras.Model):
  def __init__(self, filters, square=True, mult_wind=False, stack_wind=False, padding_offset=0, smooth_padd=False, flip_indicators=True):
    super(AdvecBlock3, self).__init__()
    self.filters_1 = filters
    self.square = square
    self.mult_wind = mult_wind
    self.stack_wind = stack_wind
    self.padding_offset = padding_offset
    self.smooth_padd = smooth_padd
    self.flip_indicators = flip_indicators

    # create padding for our convolutional layers
    self.padd_ul = ZeroPadding2D(padding=((1, 1), (1 - self.padding_offset, 1 + padding_offset)))
    self.padd_ur = ZeroPadding2D(padding=((1, 1), (1 + self.padding_offset, 1 - padding_offset)))
    self.padd_vl = ZeroPadding2D(padding=((1 - self.padding_offset, 1 + self.padding_offset), (1, 1)))
    self.padd_vr = ZeroPadding2D(padding=((1 + self.padding_offset, 1 - self.padding_offset), (1, 1)))
    # example u_l for padding_offset==-1:
    # 0 0 0 0
    # 0 0 X X
    # 0 0 X X
    # 0 0 0 0

    # if smooth_padd == False then we will use centered padding when creating smoothness indicators. we also use it for reducing dimensionality
    self.padd_center = ZeroPadding2D(padding=((1, 1), (1, 1)))

    self.ul_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ul_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ul_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ul_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.conv_out = Conv2D(self.filters_1, (1, 1), padding='valid')

    # if flip_indicators == True we will invert our smoothness indicators before feeding it into the softmax layer
    if self.flip_indicators == True:
      self.flip = Lambda(lambda b: 1/(b + K.epsilon()))

    self.mult = Multiply()

    # if square == True we will square our smoothness indicators

    if self.square == True:
      self.square = Lambda(lambda b: K.square(b))
    self.soft = Softmax(axis=-1)
  def call(self, input_tensor, training=False):
    p = input_tensor[0]
    u, v = [tf.slice(input_tensor[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)]
    # first we deal with the left boundary of the u velocity component

    # establish which padding we will be using
    padd_ul = self.padd_ul
    padd_ur = self.padd_ur
    padd_vl = self.padd_vl
    padd_vr = self.padd_vr
    # apply a convolution to the concentration field
    p_ul = tf.nn.tanh(self.ul_conv_1(padd_ul(p)))
    p_ur = tf.nn.tanh(self.ur_conv_1(padd_ur(p)))
    p_vl = tf.nn.tanh(self.vl_conv_1(padd_vl(p)))
    p_vr = tf.nn.tanh(self.vr_conv_1(padd_vr(p)))
    # apply a separate convolution to the velocity components
    ul_1 = tf.nn.tanh(self.ul_conv_2(padd_ul(u)))
    ur_1 = tf.nn.tanh(self.ur_conv_2(padd_ur(u)))
    vl_1 = tf.nn.tanh(self.vl_conv_2(padd_vl(v)))
    vr_1 = tf.nn.tanh(self.vr_conv_2(padd_vr(v)))
    # multiply the concentration and appropriate velocity component to get terms estimating mass flux
    mult_ul_1 = self.mult([p_ul, ul_1])
    mult_ur_1 = self.mult([p_ur, ur_1])
    mult_vl_1 = self.mult([p_vl, vl_1])
    mult_vr_1 = self.mult([p_vr, vr_1])
    # apply at least one convolution to this to get various weighted averages of local weighted fluxes
    x_ul_1 = tf.nn.tanh(self.ul_conv_3(padd_ul(mult_ul_1)))
    x_ur_1 = tf.nn.tanh(self.ur_conv_3(padd_ur(mult_ur_1)))
    x_vl_1 = tf.nn.tanh(self.vl_conv_3(padd_vl(mult_vl_1)))
    x_vr_1 = tf.nn.tanh(self.vr_conv_3(padd_vr(mult_vr_1)))
    # we will now try to find weights for the a_ul_1 feature maps that correspond to smoothness
    # if smooth_padd is false we will switch back to centered padding for the generation of smoothness indicators and weights
    if self.smooth_padd == False:
      padd_ul = self.padd_center
      padd_ur = self.padd_center
      padd_vl = self.padd_center
      padd_vr = self.padd_center
    # now we will estimate spatial derivatives at various orders to form the smoothness indicators
    if self.mult_wind:
      # if mult_wind is true we multiply the spatial derivatives by the proper velocity component
      mult_ul_2 = self.mult([x_ul_1, ul_1])
      x_ul_2 = tf.nn.tanh(self.ul_conv_4(padd_ul(mult_ul_2)))
      mult_ur_2 = self.mult([x_ur_1, ur_1])
      x_ur_2 = tf.nn.tanh(self.ur_conv_4(padd_ur(mult_ur_2)))
      mult_vl_2 = self.mult([x_vl_1, vl_1])
      x_vl_2 = tf.nn.tanh(self.vl_conv_4(padd_vl(mult_vl_2)))
      mult_vr_2 = self.mult([x_vr_1, vr_1])
      x_vr_2 = tf.nn.tanh(self.vr_conv_4(padd_vr(mult_vr_2)))
    else:
      x_ul_2 = tf.nn.tanh(self.ul_conv_4(padd_ul(x_ul_1)))
      x_ur_2 = tf.nn.tanh(self.ur_conv_4(padd_ur(x_ur_1)))
      x_vl_2 = tf.nn.tanh(self.vl_conv_4(padd_vl(x_vl_1)))
      x_vr_2 = tf.nn.tanh(self.vr_conv_4(padd_vr(x_vr_1)))
      # regardless of mult_wind and smooth_padd we will use a tanh layer here
    # collect this quantity (spatial derivative) at each order (i+1) to be later used for smoothness indicators

    # Now we move onto the right boundary of the u velocity component



    indicators = concatenate([x_ul_2, x_ur_2, x_vl_2, x_vr_2], axis=-1)
    # if square == True we will square the feature maps used for the smoothness indicators
    if self.square:
      indicators = self.square(indicators)
    # if flip == True we will flip the indicators to get weights
    if self.flip:
      indicators = self.flip(indicators)
    # we use a softmax layer to get normalized relative weights for the various feature maps of each velocity component
    weights = self.soft(indicators)
    # now we will collect the feature maps that we will weight
    concat_2 = concatenate([x_ul_1, x_ur_1, x_vl_1, x_vr_1], axis=-1)
    # weight the feature maps with a multiplication layer
    mult = self.mult([weights, concat_2])
    # if stack_wind == True we will include the velocity components as standalone features when forming the output
    #if self.stack_wind == True:
      #mult = concatenate([concat_2, u, v])
    out = tf.nn.tanh(self.conv_out(mult))
    return out


class AdvecBlock4(tf.keras.Model):
  def __init__(self, filters, square=True, mult_wind=False, stack_wind=False, padding_offset=0, smooth_padd=False, flip_indicators=True):
    super(AdvecBlock4, self).__init__()
    self.filters_1 = filters
    self.square = square
    self.mult_wind = mult_wind
    self.stack_wind = stack_wind
    self.padding_offset = padding_offset
    self.smooth_padd = smooth_padd
    self.flip_indicators = flip_indicators

    # create padding for our convolutional layers
    self.padd_ul = ZeroPadding2D(padding=((1, 1), (1 - self.padding_offset, 1 + padding_offset)))
    self.padd_ur = ZeroPadding2D(padding=((1, 1), (1 + self.padding_offset, 1 - padding_offset)))
    self.padd_vl = ZeroPadding2D(padding=((1 - self.padding_offset, 1 + self.padding_offset), (1, 1)))
    self.padd_vr = ZeroPadding2D(padding=((1 + self.padding_offset, 1 - self.padding_offset), (1, 1)))
    # example u_l for padding_offset==-1:
    # 0 0 0 0
    # 0 0 X X
    # 0 0 X X
    # 0 0 0 0

    # if smooth_padd == False then we will use centered padding when creating smoothness indicators. we also use it for reducing dimensionality
    self.padd_center = ZeroPadding2D(padding=((1, 1), (1, 1)))

    self.ul_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_1 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ul_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_2 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ul_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_3 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ul_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_4 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ul_conv_5 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.ur_conv_5 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vl_conv_5 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.vr_conv_5 = Conv2D(self.filters_1, (3, 3), padding='valid')
    self.conv_indicators = Conv2D(self.filters_1 * 4, (1, 1), padding='valid')
    self.conv_out = Conv2D(self.filters_1, (1, 1), padding='valid')

    # if flip_indicators == True we will invert our smoothness indicators before feeding it into the softmax layer
    if self.flip_indicators == True:
      self.flip = Lambda(lambda b: 1/(b + K.epsilon()))

    self.mult = Multiply()

    # if square == True we will square our smoothness indicators

    if self.square == True:
      self.square = Lambda(lambda b: K.square(b))
    self.soft = Softmax(axis=-1)
  def call(self, input_tensor, training=False):
    p = input_tensor[0]
    u, v = [tf.slice(input_tensor[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)]
    # first we deal with the left boundary of the u velocity component

    # establish which padding we will be using
    padd_ul = self.padd_ul
    padd_ur = self.padd_ur
    padd_vl = self.padd_vl
    padd_vr = self.padd_vr
    # apply a convolution to the concentration field
    p_ul = tf.nn.tanh(self.ul_conv_1(padd_ul(p)))
    p_ur = tf.nn.tanh(self.ur_conv_1(padd_ur(p)))
    p_vl = tf.nn.tanh(self.vl_conv_1(padd_vl(p)))
    p_vr = tf.nn.tanh(self.vr_conv_1(padd_vr(p)))
    # apply a separate convolution to the velocity components
    ul_1 = tf.nn.tanh(self.ul_conv_2(padd_ul(u)))
    ur_1 = tf.nn.tanh(self.ur_conv_2(padd_ur(u)))
    vl_1 = tf.nn.tanh(self.vl_conv_2(padd_vl(v)))
    vr_1 = tf.nn.tanh(self.vr_conv_2(padd_vr(v)))
    # multiply the concentration and appropriate velocity component to get terms estimating mass flux
    mult_ul_1 = self.mult([p_ul, ul_1])
    mult_ur_1 = self.mult([p_ur, ur_1])
    mult_vl_1 = self.mult([p_vl, vl_1])
    mult_vr_1 = self.mult([p_vr, vr_1])
    # apply at least one convolution to this to get various weighted averages of local weighted fluxes
    x_ul_1 = tf.nn.tanh(self.ul_conv_3(padd_ul(mult_ul_1)))
    x_ur_1 = tf.nn.tanh(self.ur_conv_3(padd_ur(mult_ur_1)))
    x_vl_1 = tf.nn.tanh(self.vl_conv_3(padd_vl(mult_vl_1)))
    x_vr_1 = tf.nn.tanh(self.vr_conv_3(padd_vr(mult_vr_1)))
    # we will now try to find weights for the a_ul_1 feature maps that correspond to smoothness
    # if smooth_padd is false we will switch back to centered padding for the generation of smoothness indicators and weights
    if self.smooth_padd == False:
      padd_ul = self.padd_center
      padd_ur = self.padd_center
      padd_vl = self.padd_center
      padd_vr = self.padd_center
    # now we will estimate spatial derivatives at various orders to form the smoothness indicators
    if self.mult_wind:
      # if mult_wind is true we multiply the spatial derivatives by the proper velocity component
      mult_ul_2 = self.mult([x_ul_1, ul_1])
      x_ul_2 = tf.nn.tanh(self.ul_conv_4(padd_ul(mult_ul_2)))
      mult_ur_2 = self.mult([x_ur_1, ur_1])
      x_ur_2 = tf.nn.tanh(self.ur_conv_4(padd_ur(mult_ur_2)))
      mult_vl_2 = self.mult([x_vl_1, vl_1])
      x_vl_2 = tf.nn.tanh(self.vl_conv_4(padd_vl(mult_vl_2)))
      mult_vr_2 = self.mult([x_vr_1, vr_1])
      x_vr_2 = tf.nn.tanh(self.vr_conv_4(padd_vr(mult_vr_2)))

      mult_ul_3 = self.mult([x_ul_2, ul_1])
      x_ul_3 = tf.nn.tanh(self.ul_conv_5(padd_ul(mult_ul_3)))
      mult_ur_3 = self.mult([x_ur_2, ur_1])
      x_ur_3 = tf.nn.tanh(self.ur_conv_5(padd_ur(mult_ur_3)))
      mult_vl_3 = self.mult([x_vl_2, vl_1])
      x_vl_3 = tf.nn.tanh(self.vl_conv_5(padd_vl(mult_vl_3)))
      mult_vr_3 = self.mult([x_vr_2, vr_1])
      x_vr_3 = tf.nn.tanh(self.vr_conv_5(padd_vr(mult_vr_3)))
    else:
      x_ul_2 = tf.nn.tanh(self.ul_conv_4(padd_ul(x_ul_1)))
      x_ur_2 = tf.nn.tanh(self.ur_conv_4(padd_ur(x_ur_1)))
      x_vl_2 = tf.nn.tanh(self.vl_conv_4(padd_vl(x_vl_1)))
      x_vr_2 = tf.nn.tanh(self.vr_conv_4(padd_vr(x_vr_1)))

      x_ul_3 = tf.nn.tanh(self.ul_conv_5(padd_ul(x_ul_2)))
      x_ur_3 = tf.nn.tanh(self.ur_conv_5(padd_ur(x_ur_2)))
      x_vl_3 = tf.nn.tanh(self.vl_conv_5(padd_vl(x_vl_2)))
      x_vr_3 = tf.nn.tanh(self.vr_conv_5(padd_vr(x_vr_2)))

    # collect the 'spatial derivatives' to be used for smoothness indicators

    concat = concatenate([x_ul_2, x_ur_2, x_vl_2, x_vr_2, x_ul_3, x_ur_3, x_vl_3, x_vr_3], axis=-1)

    # reduce number of channels so that the ten

    indicators = self.conv_indicators(concat)
    # if square == True we will square the feature maps used for the smoothness indicators
    if self.square:
      indicators = self.square(indicators)
    # reduce the dimensionality so we have weights for each feature map of each velocity component
    # if flip == True we will flip the indicators to get weights
    if self.flip_indicators:
      indicators = self.flip(indicators)
    # we use a softmax layer to get normalized relative weights for the various feature maps of each velocity component
    weights = self.soft(indicators)
    # now we will collect the feature maps that we will weight
    concat_2 = concatenate([x_ul_1, x_ur_1, x_vl_1, x_vr_1], axis=-1)
    # weight the feature maps with a multiplication layer
    mult = self.mult([weights, concat_2])
    # if stack_wind == True we will include the velocity components as standalone features when forming the output
    #if self.stack_wind == True:
      #mult = concatenate([concat_2, u, v])
    out = tf.nn.tanh(self.conv_out(mult))
    return out

#block = AdvecBlock3(64, num_derivs=3)
#_ = block([tf.zeros((1, 64, 64, 1)), tf.zeros((1, 64, 64, 2))])
#print(block.summary())
#exit()
# Custom advection block 2 - for input velocity vector fields stored at cell centers
def NOAA_model(block, pretrained_weights=None, input_size=(232, 396, 5)):
  inputs = Input(shape=(232, 396, 5))
  concentration = tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, 1])
  velocities = tf.slice(inputs, [0, 0, 0, 1], [-1, -1, -1, -1])
  block1 = block(filters=256)([concentration, velocities])
  block2 = block(filters=128)([block1, velocities])
  block3 = block(filters=64)([block2, velocities])
  block4 = block(filters=32)([block3, velocities])
  concat_1 = concatenate([block1, block2, block3, block4], axis=-1)
  conv_1 = Conv2D(128, 1)(concat_1)
  conv_2 = Conv2D(32, 1)(conv_1)
  conv_3 = Conv2D(8, 1)(conv_2)
  conv_4 = Conv2D(1, 1)(conv_3)
  out = Add()([conv_4, concentration])
  model = Model(inputs=inputs, outputs=out, name=str(block.__name__)+'_NOAA_format')
  model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])

  if (pretrained_weights):
    model.load_weights(pretrained_weights)

  print(model.summary())

  return model

##################
class Numerical_model(tf.keras.Model):
  def __init__(self, block, square=True, mult_wind=False, stack_wind=False, padding_offset=0, smooth_padd=False, flip_indicators=True):
    super(Numerical_model, self).__init__()
    self.block_1 = block(filters=256, square=square, mult_wind=mult_wind, stack_wind=stack_wind, padding_offset=padding_offset, smooth_padd=smooth_padd, flip_indicators=flip_indicators)
    self.block_2 = block(filters=128, square=square, mult_wind=mult_wind, stack_wind=stack_wind, padding_offset=padding_offset, smooth_padd=smooth_padd, flip_indicators=flip_indicators)
    self.block_3 = block(filters=64, square=square, mult_wind=mult_wind, stack_wind=stack_wind, padding_offset=padding_offset, smooth_padd=smooth_padd, flip_indicators=flip_indicators)
    self.block_4 = block(filters=32, square=square, mult_wind=mult_wind, stack_wind=stack_wind, padding_offset=padding_offset, smooth_padd=smooth_padd, flip_indicators=flip_indicators)
    self.conv_1 = Conv2D(8, 1)
    self.conv_2 = Conv2D(1, 1)
  def call(self, inputs):
    concentration = Lambda(lambda x: tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda x: tf.slice(x, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    block_1 = self.block_1([concentration, velocities])
    block_2 = self.block_2([block_1, velocities])
    block_3 = self.block_3([block_2, velocities])
    block_4 = self.block_4([block_3, velocities])
    concat_1 = concatenate([block_1, block_2, block_3, block_4], axis=-1)
    #conv_1 = Conv2D(128, 1)(concat_1)
    #conv_2 = Conv2D(32, 1)(conv_1)
    drop_1 = Dropout(.2)(concat_1)
    conv_3 = tf.nn.tanh(self.conv_1(drop_1))
    conv_4 = self.conv_2(conv_3)
    out = Add()([conv_4, concentration])
    return out

