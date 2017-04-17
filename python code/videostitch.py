#import modules
import os, sys
import freenect
import numpy as np
import frame_convert
import frame_convert2
import cv2
print(cv2.__version__)
import scipy.misc
import time
import timeit
import argparse
from PIL import Image as im
# from matplotlib import pyplot as plt
print('importing modules complete')
os.chdir('/Users/justinchan/Desktop/Shelves/Depth') #change to thesis directory

code = cv2.VideoWriter_fourcc('m','p','4','v')
newvid = cv2.VideoWriter()
success = newvid.open('newvid1.mp4',code,20,(640,480),True)
for i in xrange(1,554):
    nframe = cv2.imread('Shelves_disp_kinect_%d.bmp' % i)
    newvid.write(nframe)
newvid.release()