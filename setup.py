from setuptools import setup
from setuptools import find_packages
#import pip


#pip.main(['install', 'git+https://www.github.com/keras-team/keras-contrib.git'])

setup(name='covid_pipeline',
      version='0.1',
      description='Deep Learning segmentation pipeline analysis in Python',
      url='https://gitlab.com/mer17mm/mapp-coronahack',
      author='Michail Mamalakis',
      author_email='mmamalakis1@sheffield.ac.uk',
      license='GPL-3.0+',
      packages=['covid_pipeline'],
      install_requires=[
	  'niftynet',
#	  'tf-nightly-gpu',
#	  'tf-nightly',
	  'h5py',
          'numpy>=1.15.4',
          'scipy>=1.1.0',
          'matplotlib==3.0.2',
	  'dicom',
	  'pydicom==1.2.1',	
	  'opencv-python==4.0.0.21',
	  'Pillow==5.3.0',
	  'vtk>=8.1.1',	
          'future',
          'keras==2.3.0',
          'tensorflow>=2.0.0b1',
          'tensorboard>=2.0.0',
	  'tensorflow-gpu>=2.0.0b1',
	  'rq-scheduler==0.7.0',
	  'med2image',
	  'imageio',
          'gensim',
          'SimpleItk',
          'networkx',
          'sklearn',
      ],
	zip_safe=False

)


#nano ../.conda-sharc/envs/covid/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py
#add from tensorflow.python.types import core as core_tf_types
#modify def is_tensor(x):
#modify    return isinstance(x, core_tf_types.Tensor) or tf_ops.is_dense_tensor_like(x)


