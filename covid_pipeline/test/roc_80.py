#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mmamalakis1@sheffield.ac.uk

from __future__ import division, print_function
from covid_pipeline import main

# name of weights store of main segmentation
mn=main.main('covid_80')

# run train of main segmentation
mn.test_run(['denseres171','densenet121','resnet50'],'/fastdata/mer17mm/private/Data/')
# run test of main segmentation
# mn.test_run()


