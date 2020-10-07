#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from covid_pipeline import main

# name of weights store of main segmentation


mn=main.main('covid_300')
model_path= '../../../../data/mer17mm/python_source/Model_covid/weights_covid_300_denseres171.h5'                                                                          
img_path='../../../../fastdata/mer17mm/private/Data/covid_80_map/'

#first dense

layer="average_pooling2D_3"
output_path1='../../../../fastdata/mer17mm/private/Data/map_attention/dense1/'
#mn.attention_map(img_path, output_path1, model_path,layer)

#second res

#mn2=main.main('covid_300')
layer2="conv2D_2"
output_path2='../../../../fastdata/mer17mm/private/Data/map_attention/res1/'
#mn.attention_map(img_path, output_path2, model_path,layer2)

#third concate
#mn3=main.main('covid_300')
layer3='concatenate2'
output_path3='../../../../fastdata/mer17mm/private/Data/map_attention/f1/'

layerall=[layer,layer2,layer3]
output_pathall=[output_path1,output_path2,output_path3]

mn.attention_map(img_path, output_pathall, model_path,layerall)
