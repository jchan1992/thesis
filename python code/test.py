# frameworkpython needed to run matplotlib
from idlelib.PyShell import main
if __name__ == '__main__':
    main()
print('entered virtual env')
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
os.chdir('/Users/justinchan/dropbox/thesis/python') #change to thesis directory

def importfiles():
    #import test videos
    # tx1 = cv2.VideoCapture('test.mp4')
    tx2 = cv2.VideoCapture('test1.mp4')
    tx21 = cv2.VideoCapture('test2.mp4') 
    tx3 = cv2.VideoCapture('wall.mp4')
    tx31 = cv2.VideoCapture('walld.mp4')
    tx4 = cv2.VideoCapture('chairbox.mp4')
    tx41 = cv2.VideoCapture('chairboxd.mp4')
    tx5 = cv2.VideoCapture('shelves.mp4')
    tx51 = cv2.VideoCapture('shelvesd.mp4')
    tx6 = cv2.VideoCapture('hallway.mp4')
    tx61 = cv2.VideoCapture('hallwayd.mp4')
    tx = [tx2,tx3,tx4,tx5,tx6]
    td = [tx21,tx31,tx41,tx51,tx61]
    return [tx,td]

def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])
def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])
def initkinect():
    #set up kinect camera
    ctx = freenect.init()
    dev = freenect.open_device(ctx, freenect.num_devices(ctx) - 1)
    if not dev:
        freenect.error_open_device()
    return [ctx,dev]
    print('kinect setup complete')
def playkinect():
    print('display kinect stream')
    while 1:
        cv2.imshow('Depth',get_depth())
        cv2.imshow('Video',get_video())
        if cv2.waitKey(10) == 27:
            break
def stitchim2(num):
    bs = cv2.imread('result thresholding/%03dbs.jpg' % num)
    kmeans = cv2.imread('result thresholding/%03dkmeans.jpg' % num)
    kmeanso = cv2.imread('result thresholding/%03dkmeanso.jpg' % num)
    niblack10 = cv2.imread('result thresholding/%03dniblack10.jpg' % num)
    niblack100 = cv2.imread('result thresholding/%03dniblack100.jpg' % num)
    otsu = cv2.imread('result thresholding/%03dotsu.jpg' % num)
    sauv10 = cv2.imread('result thresholding/%03dsauv10.jpg' % num)
    sauv100 = cv2.imread('result thresholding/%03dsauv100.jpg' % num)
    triangle = cv2.imread('result thresholding/%03dtriangle.jpg' % num)
    gt = cv2.imread('result thresholding/%03dgt.jpg' % num)
    wellner10 = cv2.imread('result thresholding/%03dwellner10.jpg' % num)
    wellner100 = cv2.imread('result thresholding/%03dwellner100.jpg' % num)
    
    cv2.putText(bs,"Background Subtraction",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(gt,"Ground Truth",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(kmeans,"Kmeans",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(kmeanso,"Kmeans Modified",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(otsu,"Otsu",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(triangle,"Triangle",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(niblack10,"Niblack blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(niblack100,"Niblack blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(sauv10,"Sauvola blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(sauv100,"Sauvola blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(wellner10,"Wellner blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    cv2.putText(wellner100,"Wellner blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),5)
    

    height,width = bs.shape[:2]
    newim = np.zeros((height*4, width*3, 3), dtype="uint8")
    newim[0:height,0:width] = bs
    newim[height:2*height,0:width] = gt
    newim[2*height:3*height,0:width] = otsu
    newim[3*height:4*height,0:width] = triangle
    
    newim[0:height,width:2*width] = kmeans
    newim[1*height:2*height,width:2*width] = kmeanso
    newim[2*height:3*height,width:2*width] = niblack10
    newim[3*height:4*height,width:2*width] = niblack100
    
    newim[0:height,2*width:3*width] = sauv10
    newim[1*height:2*height,2*width:3*width] = sauv100
    newim[2*height:3*height,2*width:3*width] = wellner10
    newim[3*height:4*height,2*width:3*width] = wellner100

    newim[0:4*height,0:2,:] = (0,255,0)
    newim[0:4*height,4*width-2:4*width,:] = (0,255,0)
    newim[0:4*height,width-1:width+1,:] = (0,255,0)
    newim[0:4*height,2*width-1:2*width+1,:] = (0,255,0)
    newim[0:4*height,3*width-2:3*width,:] = (0,255,0)
    
    newim[0:2,0:width*3,:] = (0,255,0)
    newim[height-1:height+1,0:width*3,:] = (0,255,0)
    newim[2*height-1:2*height+1,0:width*3,:] = (0,255,0)
    newim[3*height-1:3*height+1,0:width*3,:] = (0,255,0)
    newim[4*height-2:4*height,0:width*3,:] = (0,255,0)
    

    cv2.imwrite('../%03dstitch2.jpg' % num,newim)    

# background subtraction algorithms
def basic(vid):
    print('beginning basic')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    start_time = time.time()
    # get first frame
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    # while vid.isOpened():
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,vidFrame = vid.read();
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY);
        bsFrame = cv2.absdiff(vidFrame,bgFrame);
        ret,thrFrame = cv2.threshold(bsFrame,30,255,cv2.THRESH_TRIANGLE)
        cv2.imshow('thr',thrFrame)
        cv2.waitKey(1)
    end_time = time.time();
    timet = end_time - start_time
    vid.release();
    cv2.destroyAllWindows();
    cv2.waitKey(1);
    print('finished basic');
def doubledifference(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    start_time = time.time()
    ret,bgframe = vid.read()
    videobg = bgframe[0:height,0:width]
    videobg = cv2.cvtColor(videobg,cv2.COLOR_BGR2GRAY)
    video0 = videobg.copy()

    ret,frame = vid.read()
    video = frame[0:height,0:width]
    video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=26000)):

        ret,frame1 = vid.read()
        video1 = frame1[0:height,0:width]
        video1 = cv2.cvtColor(video1,cv2.COLOR_BGR2GRAY)

        vidmask1 = cv2.absdiff(video,video0)
        vidmask2 = cv2.absdiff(video,video1)
        ret,vidmask1 = cv2.threshold(vidmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,vidmask2 = cv2.threshold(vidmask2,0,255,cv2.THRESH_TRIANGLE)
        vidmaskf = cv2.bitwise_and(vidmask1,vidmask2)
        vidmaskf = cv2.cvtColor(vidmaskf,cv2.COLOR_GRAY2BGR)
        vidout.write(vidmaskf)
        video0 = video
        video = video1
    end_time = time.time()
    timet = start_time-end_time
    vid.close()
def temporaldifferencing(vid):
    print('beginning temporal differencing')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    start_time = time.time()
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,vidFrame = vid.read();
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY);
        bsFrame = cv2.absdiff(vidFrame,bgFrame);
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        bgFrame = vidFrame
        cv2.imshow('thr',thrFrame)
        cv2.waitKey(1)
    end_time = time.time();
    timet = end_time - start_time
    vid.release();
    vidout.release();
    cv2.destroyAllWindows();
    cv2.waitKey(1);
    print('finished temporal differencing');
def runningaverage(vid):
    print('beginning running average')
    start_time = time.time()
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    avgFrame = np.float32(bgFrame)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(vidFrame,avgFrame,0.9)
        bgFrame = cv2.convertScaleAbs(avgFrame)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        cv2.imshow('vid',vidFrame)
        cv2.imshow('bg',bgFrame)
        cv2.imshow('bs',bsFrame)
        cv2.imshow('th',thrFrame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    vid.release()
def approximatemedian(vid):
    print('starting approximate median')
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY) 
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    start_time = time.time()
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        for i in xrange(height):
            for j in xrange(width):
                k = vidFrame.item(i,j)
                l = bgFrame.item(i,j) 
                if k>l:
                    bgFrame.itemset((i,j),l+1)
                elif k<l:
                    bgFrame.itemset((i,j),l-1)
        cv2.imshow('th',thrFrame)
        cv2.imshow('vid',vidFrame)
        cv2.imshow('bs',bsFrame)
        cv2.imshow('bg',bgFrame)
        cv2.waitKey(1)
        end_time = time.time()
    timet = end_time-start_time
    cv2.destroyAllWindows()
    vid.release()
def mog(vid):
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    start_time = time.time()
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,frame = vid.read()
        fgmask = fgbg.apply(frame)
        fgmask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
        cv2.imshow('frame',fgmask)
        cv2.waitKey(1)
    end_time = time.time()
    timet = end_time - start_time
    f = open('results background.txt','a')
    f.write('mog time with recording = %d\n' % timet)
    f.close()
    vid.release()
    vidout.release()
    cv2.destroyAllWindows()

def depth(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    ret,bgframe = vid.read()
    depthbg = bgframe[0:height,0:width]
    depthbg = cv2.cvtColor(depthbg,cv2.COLOR_BGR2GRAY)
    depth0 = depthbg.copy()
    start = time.time()
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,frame = vid.read()
        depth = frame[0:height,0:width]
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        
        depmask1 = cv2.absdiff(depth,depth0)
        depmask3 = cv2.absdiff(depth,depthbg)
        ret,depmask1 = cv2.threshold(depmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,depmask3 = cv2.threshold(depmask3,0,255,cv2.THRESH_TRIANGLE)
        depmaskf3 = cv2.bitwise_or(depmask1,depmask3) 
        cv2.imshow('dep4',depmaskf3)
        cv2.waitKey(1)
        depth0 = depth
    end = time.time()
    print(end-start)

def getbsframe(vid):
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    i = 0
    while(imtime<30000):
        ret,vidFrame = vid.read()
        vidFrame =cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        cv2.imwrite('%03dbs.jpg' % i,bsFrame)
        i += 1
        bgFrame = vidFrame
        imtime = vid.get(cv2.CAP_PROP_POS_MSEC) 
