# framework python needed to run matplotlib
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


# background subtraction algorithms
def basic(vid):
    print('beginning basic')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # set up video writer
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('basic.mp4',code,fps,(width,height),True)
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
        thrFrame = cv2.cvtColor(thrFrame,cv2.COLOR_GRAY2BGR)
        vidout.write(thrFrame)
        cv2.waitKey(1)
    end_time = time.time();
    timet = end_time - start_time
    vid.release();
    vidout.release();
    cv2.destroyAllWindows();
    print('finished basic');
def doubledifference(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('double difference.mp4',code,fps,(width,height),True)
    start_time = time.time()
    ret,bgframe = vid.read()
    videobg = bgframe[0:height,0:width]
    videobg = cv2.cvtColor(videobg,cv2.COLOR_BGR2GRAY)
    # videobg = cv2.equalizeHist(videobg)
    video0 = videobg.copy()
    ret,frame = vid.read()
    video = frame[0:height,0:width]
    video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    # video = cv2.equalizeHist(video)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=26000)):

        ret,frame1 = vid.read()
        video1 = frame1[0:height,0:width]
        video1 = cv2.cvtColor(video1,cv2.COLOR_BGR2GRAY)
        # video1 = cv2.equalizeHist(video1)
        vidmask1 = cv2.absdiff(video,video0)
        vidmask2 = cv2.absdiff(video,video1)
        # vidmask3 = cv2.absdiff(video,videobg)
        ret,vidmask1 = cv2.threshold(vidmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,vidmask2 = cv2.threshold(vidmask2,0,255,cv2.THRESH_TRIANGLE)
        # ret,vidmask3 = cv2.threshold(vidmask3,0,255,cv2.THRESH_TRIANGLE)
        vidmaskf = cv2.bitwise_and(vidmask1,vidmask2)
        # vidmaskf1 = cv2.bitwise_and(vidmask1,vidmask3)
        vidmaskf = cv2.cvtColor(vidmaskf,cv2.COLOR_GRAY2BGR)
        vidout.write(vidmaskf)
        # cv2.imshow('vidmask1',vidmaskf)
        # cv2.waitKey(1)
        # cv2.imshow('vidmask2',vidmaskf1)
        video0 = video
        video = video1
    end_time = time.time()
    timet = start_time-end_time
    vid.close()
    vidout.close()
def temporaldifferencing(vid):
    print('beginning temporal differencing')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # # set up video writer
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('temporal.mp4',code,fps,(width,height),True)
    start_time = time.time()
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,vidFrame = vid.read();
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY);
        bsFrame = cv2.absdiff(vidFrame,bgFrame);
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        thrFrame = cv2.cvtColor(thrFrame,cv2.COLOR_GRAY2BGR)
        bgFrame = vidFrame
        vidout.write(thrFrame)
        cv2.waitKey(1)
    end_time = time.time();
    timet = end_time - start_time
    vid.release();
    vidout.release();
    cv2.destroyAllWindows();
    print('finished temporal differencing');
def runningaverage(vid):
    print('beginning running average')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # # set up video writer
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('runningaverage.mp4',code,fps,(width,height),True)
    
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
        thrFrame = cv2.cvtColor(thrFrame,cv2.COLOR_GRAY2BGR)
        vidout.write(thrFrame)
    cv2.destroyAllWindows()
    vidout.release()
    vid.release()
def approximatemedian(vid):
    print('starting approximate median')
    #   # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # # # set up video writer
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('approximatemedian.mp4',code,fps,(width,height),True)
    
    # read input video
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY) 
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
        thrFrame = cv2.cvtColor(thrFrame,cv2.COLOR_GRAY2BGR)
        vidout.write(thrFrame)
        end_time = time.time()
    timet = end_time-start_time
    cv2.destroyAllWindows()
    vid.release()
    vidout.release()
def mog(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('mog.mp4',code,fps,(width,height),True)
    
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    start_time = time.time()
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,frame = vid.read()
        fgmask = fgbg.apply(frame)
        fgmask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
        vidout.write(fgmask)
    end_time = time.time()
    timet = end_time - start_time
    f = open('results background.txt','a')
    f.write('mog time with recording = %d\n' % timet)
    f.close()
    vid.release()
    vidout.release()
    cv2.destroyAllWindows()

def depth(vid):
        # # # set up video writer
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('depthdsdsdsd.mp4',code,fps,(width,height),True)
    
    ret,bgframe = vid.read()
    depthbg = bgframe[0:height,0:width]
    depthbg = cv2.cvtColor(depthbg,cv2.COLOR_BGR2GRAY)
    depth0 = depthbg.copy()

    while vid.isOpened():
        ret,frame = vid.read()
        depth = frame[0:height,0:width]
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        
        depmask1 = cv2.absdiff(depth,depth0)
        depmask3 = cv2.absdiff(depth,depthbg)
        ret,depmask1 = cv2.threshold(depmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,depmask3 = cv2.threshold(depmask3,0,255,cv2.THRESH_TRIANGLE)
        depmaskf3 = cv2.bitwise_or(depmask1,depmask3)
        depmaskf3 = cv2.cvtColor(depmaskf3,cv2.COLOR_GRAY2BGR)
        vidout.write(depmaskf3) 
        # cv2.imshow('dep4',depmaskf3)
        # cv2.waitKey(1)
        depth0 = depth
    vid.release()
    vidout.release()

def depth1(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    ret,bgframe = vid.read()
    depthbg = bgframe[0:height,0:width]
    depthbg = cv2.cvtColor(depthbg,cv2.COLOR_BGR2GRAY)
    depth0 = depthbg.copy()

    while(1):
        ret,frame = vid.read()
        depth = frame[0:height,0:width]
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        
        depmask1 = cv2.absdiff(depth,depth0)
        depmask3 = cv2.absdiff(depth,depthbg)
        ret,depmask1 = cv2.threshold(depmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,depmask3 = cv2.threshold(depmask3,0,255,cv2.THRESH_TRIANGLE)
        depmaskf3 = cv2.bitwise_and(depmask1,depmask3) 
        cv2.imshow('dep4',depmaskf3)
        cv2.waitKey(1)
        depth0 = depth