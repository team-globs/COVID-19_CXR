#Author: Michail Mamalakis
#Version: 0.3
#Licence:
#email:mmamalakis1@sheffield.ac.uk
#Acknowledgement: https://github.com/ece7048/cardiac-segmentation-1/tree/master/rvseg/models

from __future__ import division, print_function

from keras.layers import Input, Conv2D, Conv2DTranspose, Reshape
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Layer, Dense, Flatten, LeakyReLU,Lambda, Activation, BatchNormalization, Dropout, Dense

from tensorflow.keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras import constraints
from keras import applications
import argparse
from covid_pipeline import config, run_model
# FCN
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras_applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf




# TODO batch shape in the models when you have a big data its better for paraller analysis so extend base of  https://github.com/aurora95/Keras-FCN/blob/master/models.py

class main_net(object):

	def __init__ (self, init1, init2, height,channels,classes,width ) :
		args = config.parse_arguments()
		self.height=height
		self.channels=channels
		self.classes=classes
		self.main_activation=args.main_activation
		self.init_w=init1
		self.init_b=init2
		self.features=args.features
		self.depth=args.depth 
		self.padding=args.padding 
		self.batchnorm=args.batchnorm	
		self.dropout=args.dropout
		self.width=width
		self.temperature=1.0 
		self.max_norm_const=args.max_norm_const
		self.max_norm_value=args.max_norm_value
		self.im_length=args.height
		self.roi_shape=args.roi_shape	

###########################################ordibary U-net Euclidian ##################################
	def downsampling_block(self,input_tensor):
		_, height, width, _ = K.int_shape(input_tensor)
		print(height,width)
		assert height % 2 == 0
		assert width % 2 == 0
		# add max norm constrain
		if self.max_norm_const=='on':
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding, kernel_constraint=constraints.max_norm(self.max_norm_value))(input_tensor)
		else:
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding)(input_tensor)

		x = BatchNormalization()(x) if self.batchnorm else x
		x = Activation(self.main_activation)(x)
		x = Dropout(self.dropout)(x) if self.dropout > 0 else x
		# add max norm constrain
		if self.max_norm_const=='on':
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding, kernel_constraint=constraints.max_norm(self.max_norm_value))(x)
		else:
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding)(x)

		x = BatchNormalization()(x) if self.batchnorm else x
		x = Activation(self.main_activation)(x)
		x = Dropout(self.dropout)(x) if self.dropout > 0 else x
		return MaxPooling2D(pool_size=(2,2))(x), x

	def upsampling_block(self,input_tensor, skip_tensor):
		x = Conv2DTranspose(self.features, kernel_size=(2,2), strides=(2,2))(input_tensor)
		# compute amount of cropping needed for skip_tensor
		_, x_height, x_width, _ = K.int_shape(x)
		_, s_height, s_width, _ = K.int_shape(skip_tensor)
		print(x_height,x_width)
		print(s_height,s_width)
		h_crop = s_height - x_height
		w_crop = s_width - x_width
		assert h_crop >= 0
		assert w_crop >= 0
		if h_crop == 0 and w_crop == 0:
			y = skip_tensor
		else:
			cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
			y = Cropping2D(cropping=cropping)(skip_tensor)
		x = Concatenate()([x, y])
		# add max norm constrain
		if self.max_norm_const=='on':
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding, kernel_constraint=constraints.max_norm(self.max_norm_value))(x)
		else:
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding)(x)
		#
		x = BatchNormalization()(x) if self.batchnorm else x
		x = Activation(self.main_activation)(x)
		x = Dropout(self.dropout)(x) if self.dropout > 0 else x
		# add max norm constrain
		if self.max_norm_const=='on':
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding, kernel_constraint=constraints.max_norm(self.max_norm_value))(x)
		else:
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding)(x)
		#
		x = BatchNormalization()(x) if self.batchnorm else x
		x = Activation(self.main_activation)(x)
		x = Dropout(self.dropout)(x) if self.dropout > 0 else x
		return x

	def u_net(self):
		"""Generate U-Net model introduced in
		"U-Net: Convolutional Networks for Biomedical Image Segmentation"
		O. Ronneberger, P. Fischer, T. Brox (2015)
		Arbitrary number of input channels and output classes are supported.
		Arguments:
		height  - input image height (pixels)
		width   - input image width  (pixels)
		channels - input image features (1 for grayscale, 3 for RGB)
		classes - number of output classes (2 in paper)
		features - number of output features for first convolution (64 in paper)
		Number of features double after each down sampling block
		depth  - number of downsampling operations (4 in paper)
		padding - 'valid' (used in paper) or 'same'
		batchnorm - include batch normalization layers before activations
		dropout - fraction of units to dropout, 0 to keep all units
		Output:
		U-Net model expecting input shape (height, width, maps) and generates
		output with shape (output_height, output_width, classes). If padding is
		'same', then output_height = height and output_width = width.
		"""
		x = Input(shape=(self.height, self.width, self.channels))
		inputs = x
		skips = []
		print(self.height, self.width)
		for i in range(self.depth):

			x, x0 = self.downsampling_block(x)
			skips.append(x0)
			self.features *= 2
		# add max norm constrain
		if self.max_norm_const=='on':
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding, kernel_constraint=constraints.max_norm(self.max_norm_value))(x)
		else:
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding)(x)

		x = BatchNormalization()(x) if self.batchnorm else x
		x = Activation(self.main_activation)(x)
		x = Dropout(self.dropout)(x) if self.dropout > 0 else x
		# add max norm constrain
		if self.max_norm_const=='on':
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding, kernel_constraint=constraints.max_norm(self.max_norm_value))(x)
		else:
			x = Conv2D(self.features, kernel_size=(3,3), padding=self.padding)(x)

		x = BatchNormalization()(x) if self.batchnorm else x
		x = Activation(self.main_activation)(x)
		x = Dropout(self.dropout)(x) if self.dropout > 0 else x
		for i in reversed(range(self.depth)):
			self.features //= 2
			x = self.upsampling_block(x, skips[i])
		# add max norm constrain
		if self.max_norm_const=='on':
			x = Conv2D(filters=self.classes, kernel_size=(1,1), padding=self.padding, kernel_constraint=constraints.max_norm(self.max_norm_value))(x)
		else:
			x = Conv2D(filters=self.classes, kernel_size=(1,1), padding=self.padding)(x)

		logits = Lambda(lambda z: z/self.temperature)(x)

		probabilities = Activation('softmax')(logits)		
		return Model(inputs=inputs, outputs=probabilities)


	def deep_conv(self,input_shape=(64, 64)):
#TODO generilized input structure DL
		"""
		"""
		depth=input_shape[0]
		up_depth=2*depth
		model = Sequential()
		model.add(Conv2D(depth, (11,11), strides=(1, 1), padding='valid', activation=self.main_activation, input_shape=(input_shape[0], input_shape[1], 1)))
		model.add(AveragePooling2D((2,2)))
		model.add(Conv2D(up_depth, (10, 10), strides=(1, 1), padding='valid',activation=self.main_activation ))
		model.add(AveragePooling2D((2,2)))
		model.add(Reshape([-1, 128*9*9]))
		model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(Reshape([-1, self.roi_shape, self.roi_shape]))
		if self.init_w!=[]:
			F=np.reshape(self.init_w,(11,11,1,depth))
			weights=[]
			weights.append(F)
			weights.append(self.init_b)
			print(F.shape, self.init_b.shape)
			model.layers[0].set_weights(weights)
		return model



