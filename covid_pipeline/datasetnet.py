#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk
from __future__ import division, print_function
import glob
import matplotlib.patches as patches
import json
import numpy as np
from matplotlib.path import Path
import pydicom
import pydicom.uid
import dicom
import cv2
import matplotlib.pyplot as plt
import os
import scipy.misc
from PIL import Image
from PIL.Image import fromarray
import tensorflow as tf
import vtk
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
from covid_pipeline import config
import argparse
from keras import utils
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.uid import generate_uid
from PIL import Image, ImageDraw
#from med2image import med2image


class datasetnet:

#####################################
##### INITIALIZATION ##############
####################

	def __init__ (self,analysis,path_case='main') :
		"""
		Initializare of the config file

		"""
		args = config.parse_arguments()
		self.path_case=path_case
		self.rotated="False"
		self.channels=args.channels
		self.channels2=args.classes	
		if self.path_case=='main':
			self.image_shape=args.image_shape
			self.original_image_shape=args.original_image_shape
			self.roi_shape=args.roi_shape
			self.data_path=args.store_data_test
			self.data_path2=args.datapath
			self.STORE_TXT=args.store_txt
			self.counter_path='/contour/'
			self.data_extention = args.data_extention
			self.counter_extention = args.counter_extention
			self.PATH_IMAGES=''
			self.PATH_IMAGES2=''

		# seperate the train of ROI with the train set for the u_net. Thus take the epi and endo seperate in u_net and train the ROI detection in both
		if (analysis=='train' or analysis=='test') :		
			self.n_set_pre=analysis
			self.n_set=analysis
		else:
			self.n_set_pre='train_prediction'
			self.n_set='train'

		self.patient_list=args.patient_list
		self.store_contour=args.store_data_test
		self.image_part = np.zeros([self.original_image_shape,self.original_image_shape,self.channels])
		self.shuffle=args.shuffle
		self.num_preprocess_threads=args.num_cores
		self.batch_size=args.batch_size
		self.STORE_PATH=args.store_data_test
		self.type_analysis=args.type_analysis



#####################################
##### BASE FUNCTION OF CLASS############
####################


	def create_dataset(self):
		"""
		Creating the dataset from the images and the contour for the CNN-unet-algorithm.
		
		mask: the matrix of labels in txt no visible (exception in main is the same with cmask)
 		cmask: the matrix of labels in image version visible (polygon version)

		"""  
		# Create dataset from determine json file
		series = json.load(open(self.STORE_TXT+'/'+self.patient_list))[self.n_set]

		if series=={}:
			print('The %s is empty...' %self.n_set)
			X, Y, cmask_total=[],[],[]
		else:
			self.count1=0
			self.count3=0
			# call the way that the data are stored
			images_total,images_total_full,mask_total,cmask_total=self.quick_use_data(series)
			# reshape the outpouts
			X = np.reshape(np.array(images_total), [len(images_total), self.image_shape, self.image_shape, self.channels])
			# store the masks
			Y=np.array(mask_total)
			Xtotal=np.array(images_total_full)
			if self.type_analysis=='SE':
				Y = np.reshape(np.array(mask_total), [len(mask_total), mask_total.shape[3], self.roi_shape, self.roi_shape])
			Xtotal = np.reshape(np.array(images_total_full), [len(images_total_full), self.original_image_shape, self.original_image_shape, self.channels])

		print('Dataset shape :', X.shape, Y.shape)
                        
                        
		return X,Xtotal, Y, cmask_total, self.STORE_TXT



	def load_images(self,contours_list_series,case=None,serie=None):
		"""
		load images base of the path dcm or nii, png, jpg
		"""
		images=[]
		images_fullsize=[]
		#print(len(contours_list_series))
		for o in range(0,len(contours_list_series)):	
			c=contours_list_series[o]
			# Get images path base the store style for ROI	
			image_path=c
			# open image as numpy array 
			if self.data_extention=='jpeg' or self.data_extention=='png' or self.data_extention=='tif' or self.data_extention=='jpg':
				#print(image_path)     
				if self.channels==1:                                  
					image_part_store = Image.open(image_path).convert('L')
				else:
					image_part_store = Image.open(image_path).convert('RGB')
				#print(image_part_store.bits) 
				self.image_part= np.array(image_part_store)
				#print(self.image_part.shape)
			elif self.data_extention=='dcm':
				# dicom and pydicom call 
				try:
					if self.channels==1:
						self.image_part = dicom.read_file(image_path).pixel_array
					else:
						self.image_part = dicom.read_file(image_path).pixel_array					
					#print("normal dicom")

				except dicom.errors.InvalidDicomError as exc:
					try:
						dcm_file = pydicom.dcmread(image_path)
						self.image_part =dcm_file.pixel_array.astype(float)
						#print("normal pydicom")
				
					except pydicom.errors.InvalidDicomError as exc:				
						
						print("general exception dicom")
						self._dcm = dicom.read_file(image_path,force=True)
						image = self._dcm.pixel_array
						self.image_part=imageprint(self.image_part.shape)
					except:
						pass
					finally:
						pass 
				finally:
					pass
				#self.image_part= ArrayDicom
			else:
				print("Please define the extension of image data 'dcm' 'jpeg' ")
			#check for rotate image
			height, width = np.array(self.image_part).shape[0],np.array(self.image_part).shape[1]
			if width < height:
				np.rot90(np.array(self.image_part)) 
				self.rotated="True"
			else:
				self.image_part=self.image_part
			#resize
			if (self.type_analysis=='SE'):
				image_p=cv2.resize((self.image_part), (self.image_shape, self.image_shape),interpolation=cv2.INTER_NEAREST)
				full_image=cv2.resize((self.image_part), (self.original_image_shape, self.original_image_shape),interpolation=cv2.INTER_NEAREST)
				image_p=np.reshape(np.array(image_p),[ 1, self.channels, np.array(image_p).shape[0], np.array(image_p).shape[1]])
				full_image=np.reshape(np.array(full_image),[1, self.channels, np.array(full_image).shape[0], np.array(full_image).shape[1]])
			if (self.type_analysis=='CL'):
				image_p=np.reshape(np.array(self.image_part),[ self.channels*np.array(self.image_part).shape[0], np.array(self.image_part).shape[1]])
				full_image=np.reshape(np.array(self.image_part),[ self.channels*np.array(self.image_part).shape[0], np.array(self.image_part).shape[1]])
				image_p=cv2.resize(image_p, (self.channels*self.image_shape, self.image_shape),interpolation=cv2.INTER_NEAREST)
				full_image=cv2.resize(full_image, (self.channels*self.original_image_shape, self.original_image_shape),interpolation=cv2.INTER_NEAREST)
				image_p=np.reshape(np.array(image_p),[ 1, self.channels, self.image_shape, self.image_shape])	
				full_image=np.reshape(np.array(full_image),[1, self.channels, self.original_image_shape, self.original_image_shape])

			if o==0:
				images=image_p
				images_fullsize=full_image
				#print(images.shape)
			if o!=0:
	
				images=np.append(image_p, images, axis=0) 
				images_fullsize=np.append(full_image, images_fullsize, axis=0)
			#print(len(np.array(images).shape))
			
			if (len(np.array(images).shape)==4 and self.type_analysis=='SE'): 
				images=np.reshape(np.array(images),[len(np.array(images))*np.array(images).shape[3], np.array(images).shape[1], np.array(images).shape[2]])
				images_fullsize=np.reshape(np.array(images_fullsize),[len(np.array(images_fullsize))*np.array(images_fullsize).shape[3], np.array(images_fullsize).shape[1], np.array(images_fullsize).shape[2]])
			#print(np.array(images).shape) ecoAng$osasasa

         
		self.count1=self.count1+1
		#case without dependence of mask with image in data slices etc)
		return images,images_fullsize

	def load_masks(self,contours_list_series,contour_path_base=None,label=None,folder=0):
		"""
		load masks base of the path txt, vtk, jpeg, png

		"""
		Ymask=np.zeros((1,2))
		contours_jpeg, contours,contours_mask  = [],[],[]
		#print(contours_list)
		
		for o in range(0,len(contours_list_series)):
			c=contours_list_series[o]
			#print('counder list :', c)
			# Get contours and images path
			#idx_contour = contours_list_series.index(c)
			#print(idx_contour)	
			each_contour_path=c
			if (self.type_analysis=='SE'):					
				if self.counter_extention=='txt':
					contour,count_mask=self.txt_converter(each_contour_path)
					contours.append(contour)
					contours_mask.append(count_mask)
				elif self.counter_extention=='vtk':
					contour,count_mask=self.vtk_converter(each_contour_path)
					contours.append(contour)
					contours_mask.append(count_mask)
				elif self.counter_extention=='jpeg' or self.counter_extention=='png'or self.data_extention=='tif' or self.data_extention=='jpg':
					contour_mask_store = Image.open(each_contour_path).convert('L')
					contours_mask.append(cv2.resize(np.array(contour_mask_store), (self.original_image_shape, self.original_image_shape),interpolation=cv2.INTER_NEAREST) )
					contours_jpeg.append(cv2.resize(np.array(contour_mask_store), (self.roi_shape, self.roi_shape),interpolation=cv2.INTER_AREA) )    				
                                
                                # dicom and pydicom call
				elif self.counter_extention=='dcm':
					try:
						contour_mask_store= dicom.read_file(each_contour_path).pixel_array
						contours_mask.append(cv2.resize(np.array(contour_mask_store), (self.original_image_shape, self.original_image_shape),interpolation=cv2.INTER_NEAREST) )
						contours_jpeg.append(cv2.resize(np.array(contour_mask_store), (self.roi_shape, self.roi_shape),interpolation=cv2.INTER_AREA) )
                                        
					except dicom.errors.InvalidDicomError as exc:
						try:
							dcm_file = pydicom.dcmread(each_contour_path)
							contour_mask_store =dcm_file.pixel_array.astype(float)
							contours_mask.append(cv2.resize(np.array(contour_mask_store), (self.original_image_shape, self.original_image_shape),interpolation=cv2.INTER_NEAREST) )
							contours_jpeg.append(cv2.resize(np.array(contour_mask_store), (self.roi_shape, self.roi_shape),interpolation=cv2.INTER_AREA) )
                                                
                                                       
						except pydicom.errors.InvalidDicomError as exc:
							print("general exception dicom")
							in_dcm = dicom.read_file(each_contour_path,force=True)
							contour_mask_store = in_dcm.pixel_array
							contours_mask.append(cv2.resize(np.array(contour_mask_store), (self.original_image_shape, self.original_image_shape),interpolation=cv2.INTER_NEAREST) )
							contours_jpeg.append(cv2.resize(np.array(contour_mask_store), (self.roi_shape, self.roi_shape),interpolation=cv2.INTER_AREA) )
							print(self.image_part.shape)
						except:
							pass
						finally:
							pass
					finally:
						pass
				else:
					print("Please define the extension of mask data 'dcm' 'txt' 'jpeg' 'vtk' ")

				if self.counter_extention=='dcm' or self.counter_extention=='jpeg' or self.counter_extention=='png'or self.data_extention=='tif' or self.counter_extention=='jpg':
					Ymask = (contours_jpeg)
				else:
					Ymask = (contours)				
				Ymask= np.reshape(np.array(Ymask,dtype='uint8'), [len(Ymask), self.roi_shape, self.roi_shape, self.channels2])
				contours_mask=np.reshape(np.array(contours_mask,dtype='uint8'), [len(contours_mask), self.original_image_shape, self.original_image_shape,  self.channels2])

			elif (self.type_analysis=='CL'):
				contour_mask_store =np.zeros((self.channels2), dtype=int)
				contour_mask_store[folder]=1
				contours_mask.append(contour_mask_store) 
				contours_jpeg.append(contour_mask_store)    				 

				if self.counter_extention=='dcm' or self.counter_extention=='jpeg' or self.counter_extention=='png'or self.data_extention=='tif' or self.counter_extention=='jpg':
					Ymask = (contours_jpeg)
				else:
					Ymask = (contours)

				#Ymask= convert_to_one_hot(Ymask, self.channels2).T
				#contours_mask=convert_to_one_hot(contours_jpeg, self.channels2).T
			else:
				print("Please define type analysis in config file 'SE' or 'CL' ")	
			       
			self.count3=self.count3+1
		Ymask2=Ymask
		return Ymask2,contours_mask


#####################################
#####STORE STYLE##############
####################


	def quick_use_data(self,series):
		"""
		quick use of small random dataset no dependence image with masks according the path.(mask file inside the image file of the patient)

		"""
		images_total,images_total_full,mask_total,cmask_total=[],[],[],[]
		o=0
		folder=0
		for case, serie in series.items():  
			contour_path_base = self.data_path + '/%s' % (case)
			print(contour_path_base)
			contours_list_single=[]			
			contours_list = sorted(glob.glob(contour_path_base +'/*.%s' %self.data_extention)) 
			#print(contours_list)
			if self.type_analysis=='CL':
				contours_list2 = sorted(glob.glob(contour_path_base + '/*.%s' %self.data_extention))
			else:
				contours_list2 = sorted(glob.glob(contour_path_base + self.counter_path+'*.%s' %self.counter_extention))# need modification does not take in series but it takes the data randomly so different from images
			#print(contours_list2)
			#contours_list_single.append(contours_list[0])
			images,images_full= self.load_images(contours_list)
			mask,cmask= self.load_masks(list(contours_list2),folder=folder)
			if(o==0):
				cmask_total=cmask
				mask_total=mask
				images_total=images
				images_total_full=images_full
			print(np.array(images).shape)
			print(np.array(images_total).shape)
			print(np.array(mask).shape)
			print(np.array(mask_total).shape)
			if(o!=0):
				mask_total=np.append(mask,mask_total,axis=0)
				cmask_total=np.append(cmask,cmask_total,axis=0)
				images_total=np.append(images,images_total,axis=0)
				images_total_full=np.append(images_full,images_total_full,axis=0)
			folder=folder+1
			o=o+1	
		Y_total=np.array(mask_total)
		cY_total=np.array(cmask_total)
		X_total=np.array(images_total)
		cX_total=np.array(images_total_full)
		print(X_total.shape,cX_total.shape)
		print(Y_total.shape,cY_total.shape)
		if self.type_analysis=='SE':
			X_total=np.reshape(X_total, [X_total.shape[0], X_total.shape[1], X_total.shape[2], self.channels])
			cX_total=np.reshape(cX_total, [cX_total.shape[0], cX_total.shape[1], cX_total.shape[2],  self.channels])
			Y_total=np.reshape(Y_total, [Y_total.shape[0], Y_total.shape[1], Y_total.shape[2], self.channels2])
			cY_total=np.reshape(cY_total, [cY_total.shape[0], cY_total.shape[1], cY_total.shape[2],  self.channels2])
			if(len(X_total.shape)==5):
				X_total=np.reshape(X_total, [X_total.shape[0]*X_total.shape[1], X_total.shape[2], X_total.shape[3],  self.channels])
				cX_total=np.reshape(cX_total, [cX_total.shape[0]*cX_total.shape[1], cX_total.shape[2], cX_total.shape[3],  self.channels])
				Y_total=np.reshape(Y_total, [Y_total.shape[0]*Y_total.shape[1], Y_total.shape[2], Y_total.shape[3],  self.channels2])
				cY_total=np.reshape(cY_total, [cY_total.shape[0]*cY_total.shape[1], cY_total.shape[2], cY_total.shape[3],  self.channels2])
			print(X_total.shape,cX_total.shape)
			print(Y_total.shape,cY_total.shape)

		return X_total,cX_total,Y_total,cY_total



#####################################
##### Extention OF the Mask DATA##############
####################

	def txt_converter(self,each_contour_path):
		"""
		Convert the contour from txt extention to jpg

		"""

		contours,contour_mask=[],[]

		if self.path_case=='main':
			x, y = np.loadtxt(each_contour_path).T
			if self.rotated=="True":
				x, y = y, self.image_height - x
			BW_8BIT = 'L'
			polygon = list(zip(x, y))
			image_dims = (self.original_image_shape, self.original_image_shape)
			img = Image.new(BW_8BIT, image_dims, color=0)
			ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
			norm=1
			contour_mask.append(norm * np.array(img, dtype='uint8'))
			contours.append(norm * np.array(img, dtype='uint8'))

		return  contours ,contour_mask
	
	def vtk_converter(self,each_contour_path):
		"""
		Convert the contour from vtk extention to jpg

		"""
		contours,contour_mask=[],[]
		vtkarray = self.GetPointData().GetArray(each_contour_path)
		if vtkarray:
			array = vtk_to_numpy(vtkarray)
		if array.dtype == np.int8:
			array = array.astype(np.bool) 
		if self.path_case=='main':
			x, y = np.loadtxt(each_contour_path).T
			if self.rotated=="True":
				x, y = y, self.image_height - x
			BW_8BIT = 'L'
			polygon = array
			image_dims = (self.original_image_shape, self.original_image_shape)
			img = Image.new(BW_8BIT, image_dims, color=0)
			ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
			norm=1
			contour_mask.append(norm * np.array(img, dtype='uint8'))
			contours.append(norm * np.array(img, dtype='uint8'))
		return  contours,contour_mask

