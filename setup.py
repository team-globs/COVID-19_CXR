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
          'seaborn',
          'h5py',
          'numpy>=1.15.4',
          'scipy>=1.1.0',
          'matplotlib>=3.1.0',
          'dicom',
          'pydicom>=2.1.1',
          'opencv-python>=4.0.0.21',
          'Pillow==8.1.2',
          'vtk>=7.2.0',
          'future',
          'keras==2.4.1',
          'tensorflow>=2.4.1',
          'tensorboard>=2.4.1',
          'tensorflow-gpu>=2.4.1',
          'rq-scheduler>=0.7.0',
          'med2image',
          'imageio',
          'gensim',
          'SimpleItk',
          'networkx',
          'sklearn',

      ],
	zip_safe=False

)




