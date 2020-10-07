#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk


from __future__ import division, print_function

import numpy as np
from keras.layers import Lambda, Input, Conv2D, Concatenate, AveragePooling2D, AveragePooling1D, Dense, Flatten, Reshape, Activation, Dropout, Dense
from keras.models import Model
from covid_pipeline import regularization, config, main_net
from keras.applications import VGG16, VGG19, MobileNet, ResNet50, DenseNet121, DenseNet169, DenseNet201 #vgg16, vgg19, resnet_v1, resnet_v2, densenet, mobilenetv2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
import logging

class create_net:

	def net(self, init1, init2, case, height, channels, classes, width):
		
		name=case
		if self.channels==3:
			init_weights='imagenet'
			i_t=False
			pool=None
			cl=1000
		else:
			init_weights=None
			i_t=True
			pool=None
			cl=2000

		if self.main_model=="vgg16":
			init_model=VGG16(include_top=i_t, weights=init_weights,input_shape=(height, width,self.channels))
			model=self.fine_tuned(init_model)
		elif self.main_model=="vgg19":
			init_model=VGG19(include_top=i_t, weights=init_weights,input_shape=(height, width,self.channels))
			model=self.fine_tuned(init_model)
		elif self.main_model=="resnet50":
			init_model=ResNet50(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet			
			model=self.fine_tuned(init_model)
		elif self.main_model=="resnet101":
			init_model=ResNet101(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet
			model=self.fine_tuned(init_model)
		elif self.main_model=="resnet152":
			init_model=ResNet152(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet
			model=self.fine_tuned(init_model)
		elif self.main_model=="densenet121":
			init_model=DenseNet121(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels),pooling=pool,classes=cl) #weights initial imagenet
			model=self.fine_tuned(init_model)
		elif self.main_model=="densenet169":
			init_model=DenseNet169(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels),pooling=pool,classes=cl) #weights initial imagenet
			model=self.fine_tuned(init_model)
		elif self.main_model=="densenet201":
			init_model=DenseNet201(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels),pooling=pool,classes=cl) #weights initial imagenet
			model=self.fine_tuned(init_model)
		elif self.main_model=="mobilenet":
			init_model=MobileNet(include_top=i_t, weights=init_weights, input_shape=(height, width,self.channels)) #weights initial imagenet
			model=self.fine_tuned(init_model)
		elif self.main_model=='denseres171':
			init_weights=np.array([str(self.store_model + "/weights_" + name + "_resnet50.h5"), str(self.store_model + "/weights_" + name+ "_densenet121.h5")])
			model=self.denseres_171(i_t, init_weights, pool,cl, height, width)
		else:			
			print('Error: no main model file')

		if self.load_weights_main:
			logging.info("Load main weights from: {}".format(self.store_model+self.load_weights_main))
			model.load_weights(self.store_model+self.load_weights_main)


		for p in model.layers:
			print(p.name.title(), p.input_shape, p.output_shape)


		return model		

	def __init__ (self,model) :
		args = config.parse_arguments()
		self.main_model=model
		self.load_weights_main=args.load_weights_main		
		self.store_model=args.store_txt
		self.batch_size=args.batch_size
		self.channels=args.channels
		self.num_classes=args.classes

        #preprocessing layer
	
	def fine_tuned(self,pretrained_model):
		for p in pretrained_model.layers:
			print(p.name.title(), p.input_shape, p.output_shape)
		new_DL=pretrained_model.output
		new_DL=Flatten()(new_DL)
		new_DL=Dense(1024, activation="relu")(new_DL)	#64
		new_DL=Dropout(0.3)(new_DL)
		new_DL=Dense(512, activation="relu")(new_DL)	#64
		new_DL=Dropout(0.3)(new_DL)	
		new_DL=Dense(self.num_classes, activation="sigmoid")(new_DL) #2
		#new_DL=Reshape([2,2,1])(new_DL)
		#print(new_DL)
		return Model(inputs=pretrained_model.input, outputs=new_DL)

	def denseres_171(self,i_t, init_weights, pool,cl,height, width):

		input=Input(shape=(height, width,self.channels))
		res=ResNet50(include_top=i_t, weights='imagenet', input_shape=(height, width,self.channels))
		res=self.fine_tuned(res)
		dense=DenseNet121(include_top=i_t, weights='imagenet', input_shape=(height, width,self.channels), pooling=pool,classes=cl)
		dense=self.fine_tuned(dense)
		print('res: ')
		res.load_weights(init_weights[0])
		dense.load_weights(init_weights[1])

		res56=res.get_layer("activation_10").output
#		res56=Conv2D(128, (3, 3), padding="same", activation="relu")(res56)
#		res56=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(res56)
		dense56=dense.get_layer("pool2_relu").output
		#concatenate_layer56 = Concatenate(axis=-1)([res56,dense56])
		#concatenate_layer56 = Lambda(self.concat, name="concatenate1" )([res56, dense56])

		res28=res.get_layer("activation_22").output
		dense28=dense.get_layer("pool3_relu").output
		res14=res.get_layer("activation_40").output
		dense14=dense.get_layer("pool4_relu").output
		res7=res.get_layer("activation_49").output
		dense7=dense.get_layer("relu").output

		reso=Model(inputs=res.input,outputs=[res56,res28,res14,res7])
		denseo=Model(inputs=dense.input,outputs=[dense56,dense28,dense14,dense7])

		resm=reso(input)
		densem=denseo(input)

		concatenate_layer56 = Lambda(self.concat, name="concatenate1" )([resm[0], densem[0]])
#
#		res28=res.get_layer("activation_22").output
#		res28=Conv2D(256, (3, 3), padding="same", activation="relu")(res28)
#		res28=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(res28)
#		dense28=dense.get_layer("pool3_relu").output
#		#concatenate_layer28 = Lambda(self.concat, name="concatenate2" )([res28, dense28])

#		res28m=Model(inputs=res.input,outputs=[res28])
#		dense28m=Model(inputs=dense.input,outputs=[dense28])
		concatenate_layer28 = Lambda(self.concat, name="concatenate2" )([resm[1], densem[1]])

#		res14=res.get_layer("activation_40").output	
#		res14=Conv2D(512, (3, 3), padding="same", activation="relu")(res14)
#		res14=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(res14)
#		dense14=dense.get_layer("pool4_relu").output
#		#concatenate_layer14 = Lambda(self.concat, name="concatenate3" )([res14, dense14])

#		res14m=Model(inputs=res.input,outputs=[res14])
#		dense14m=Model(inputs=res.input,outputs=[dense14])
		concatenate_layer14 = Lambda(self.concat, name="concatenate3" )([resm[2], densem[2]])

#		res7=res.get_layer("activation_49").output
#		res7=Conv2D(1028, (3, 3), padding="same", activation="relu")(res7)
#		res7=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(res7)
#		dense7=dense.get_layer("relu").output
#		#concatenate_layer7 = Concatenate(axis=-1)([res7,dense7])
#		#concatenate_layer7 = Lambda(self.concat, name="concatenate4" )([res7, dense7])

#		res7m=Model(inputs=res.input,outputs=[res7])
#		dense7m=Model(inputs=dense.input,outputs=[dense7])
		concatenate_layer7 = Lambda(self.concat, name="concatenate4" )([resm[3], densem[3]])
		#vgg connection of the features of conc

		c11=Conv2D(512, (3, 3), padding="same", activation="relu")(concatenate_layer56)	
		c21=Conv2D(1024, (3, 3), padding="same", activation="relu")(concatenate_layer28)

		c31=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer14)
		a31=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c31)

		c4=Conv2D(2048, (3, 3), padding="same", activation="relu")(concatenate_layer7)
		a4=AveragePooling2D((2, 2), strides = (1, 1), padding = "same")(c4)


		a11=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c11)
		c12=Conv2D(1024, (2, 2), padding="same", activation="relu")(a11)
		a12=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c12)
		c13=Conv2D(2048, (2, 2), padding="same", activation="relu")(a12)
		a13=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c13)

		a21=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c21)
		c22=Conv2D(2048, (2, 2), padding="same", activation="relu")(a21)
		a22=AveragePooling2D((2, 2), strides = (2, 2), padding = "same")(c22)

		#concatenate_f1 = Concatenate(axis=-1)([a13,a22])
		#concatenate_f2 = Concatenate(axis=-1)([a31,a4])
		#concatenate_f3 = Concatenate(axis=-1)([concatenate_f2, concatenate_f1])

		concatenate_f1 = Lambda(self.concat, name="concatenatef1" )([a13, a22])
		concatenate_f2 = Lambda(self.concat, name="concatenatef2" )([a31, a4])
		concatenate_f3 = Lambda(self.concat, name="concatenatef3" )([concatenate_f1, concatenate_f2])

		DL = concatenate_f3
		new_DL=Flatten(name="f2")(DL)
		new_DL=Dense(1024, activation="relu", name="dense1")(new_DL)   #64
		new_DL=Dropout(0.3)(new_DL)
		new_DL=Dense(512, activation="relu", name="dense2")(new_DL)    #64
		new_DL=Dropout(0.3)(new_DL)
		new_DL=Dense(self.num_classes, activation="sigmoid", name="dense3")(new_DL) #2
                #new_DL=Reshape([2,2,1])(new_DL)
                
		return Model(inputs=input, outputs=new_DL)

	def concat(self,input_list):
		height=input_list[0]
		weight=input_list[1]
		x=Concatenate(axis=-1)([height,weight])
		return x
