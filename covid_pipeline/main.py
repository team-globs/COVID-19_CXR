#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
import numpy as np
import sys
import os
import cv2
import argparse
import logging
import tensorflow as tf
from covid_pipeline import  create_net, run_model, config, datasetnet, store_model, handle_data
from covid_pipeline import main_net
from pylab import *
from covid_pipeline import regularization
from covid_pipeline.regularization import data_augmentation
import time
from keras import utils
from PIL import Image
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt

class main(object):

	def __init__ (self, mmn) :
		args = config.parse_arguments()
		self.width=args.width
		self.height=args.height
		self.channels=args.channels
		self.classes=args.classes
		self.loss_main=args.loss_main
		self.main_model_name=mmn
		self.batch_size=args.batch_size
		self.original_image_shape_roi=args.original_image_shape_roi
		self.original_image_shape_main=args.original_image_shape
		self.data_augm=args.data_augm
		self.data_augm_classic=args.data_augm_classic
		self.store_model_path=args.store_txt
		self.validation=args.validation	
		self.main_activation=args.main_activation
		self.data_extention=args.data_extention
		self.counter_extention=args.counter_extention	
		self.type_analysis=args.type_analysis
		self.main_model=args.main_model
		self.mnm=mmn
		 
##################################################################################################################################
	def train_run(self):

		'''
			Run model create the main segmentation model of image
		'''
		#TODO tine learning-train model tha already stored with weigths$
		#Main-train
		#Crop the label of the myo in ROI shape to learn the main model
		#cretate dataset

		dsn=datasetnet.datasetnet('train')
		X, X_total, Y, contour_mask, storetxt = dsn.create_dataset()
		cn=create_net.create_net(self.main_model)
		Xaug, Yaug= [], []
		classes=1

		if ((self.data_augm == 'True') or (self.data_augm_classic == 'True')):
			for i in data_augmentation(X,Y):
				Xaug.append(i[0])
				Yaug.append(i[1])
			image, mask= np.asarray(Xaug) , np.asarray(Yaug)
			_, _, height, width, channels = image.shape
			print(mask.shape)
			if (self.type_analysis=='SE'):
				_, _, classes, _ ,_ = mask.shape
			if (self.type_analysis=='CL'):
				_,_, classes= mask.shape

		else:

			image, mask= np.asarray(X) , np.asarray(Y)
			print(image.shape, mask.shape)
			_, height, width, channels = image.shape
			if (self.type_analysis=='SE'):
				_, classes, _ ,_ = mask.shape
			if (self.type_analysis=='CL'):
				_, classes= mask.shape

		print(height,width,channels,classes)
		#create the analysis net
		model_structure=cn.net([], [], self.main_model_name , height,channels,(classes),width) # classes +1 to take into account the background
		rm=run_model.run_model(self.main_model_name)
		rm.weight_name=self.main_model
		print(X.shape,Y.shape)
		model_weights, history_main =rm.run_training(self.loss_main, model_structure, X, Y)


	def test_run(self,model_name,path):
	
		'''
			Test model to create the main segmentation model of image
		'''	
		sm=store_model.store_model()
		dsn_main_test=datasetnet.datasetnet('train')
		mainXtest2,  mainX_total2,mainY2, maincontour_mask2, mainstoretxt2 = dsn_main_test.create_dataset()
		mainXtest=mainX_total2
		
		if self.type_analysis=='CL':
			colors = cycle(['aqua', 'darkorange', 'cornflowerblue','viridis', 'plasma', 'inferno', 'magma', 'cividis'])
			for o in range(3): 
				plt.figure()
				fpr, tpr, roc_auc=sm.load_best_callback(mainXtest,mainY2, model_name[o], self.mnm)
				for i, color in zip(range(self.classes), colors):
					plt.plot(fpr[i], tpr[i],label='ROC curve of network %s of class %s (area = %0.001f)'  %(o, i, roc_auc[i]))
				plt.plot([0, 1], [0, 1], 'k--')
				plt.xlim([0.0, 1.0])
				plt.ylim([0.0, 1.05])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Some extension of Receiver operating characteristic to multi-class of %s' %model_name[o])
				plt.legend(loc="lower right")
				plt.savefig(path+'ROC_curve_%s.png' % model_name[o])


		if self.type_analysis=='SE':

			if mainXtest2==[]:
				print(" no testing ... ")
				print(" Continue ... ")
			else:
				mainpred2, mainmetric2=sm.load_best_callback(mainXtest,mainY2,self.main_model_name,self.main_model_name)
				# store images 
				vo=vision_outputs('main')
				vo.store_test_results(mainmetric2)
				print('Final test loss:', (mainmetric2[0]))
				print('Final test accuracy:', (mainmetric2[1]*100.0))
				print('Final test accuracy:', (mainmetric2[2]*100.0))
				print('Final test accuracy:', (mainmetric2[3]*100.0))
				print(mainpred2.shape)
				# store images 
				length2=len(mainXtest2)
				classes2=self.classes


	

	def load_img_from_fol(self,folder):
		images = []
		for filename in os.listdir(folder):
			img = Image.open(os.path.join(folder,filename)).convert('RGB')
			images.append(img)
		return images

		
	
	def attention_map(self,img_path,output_path,model_path,layer):
		folder=img_path
		images= []
		o=0
		for filename in os.listdir(folder):
			print(filename)
			img=Image.open(os.path.join(folder,filename)).convert('L')
			img=np.array(img)
			img=np.resize(np.array(img),[self.channels, self.height, self.width])
			img=np.reshape(np.array(img),[1, self.channels, self.height, self.width])			
			if o==0:
				images=img
				o=1
			else:
				images=np.append(img,images,axis=0)
		images=np.array(images)
		print(images.shape)
		hd=handle_data.handle_data(images,images,self.main_model_name)
		hd.visualize_class_activation_map(model_path, output_path,layer)


