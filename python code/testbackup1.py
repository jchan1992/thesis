from idlelib.PyShell import main
if __name__ == '__main__':
    main()
print('entered virtual env')
#basic setup
#import modules
import os, sys
import freenect
import matplotlib
import numpy as np
import frame_convert
import frame_convert2
import cv2
print(cv2.__version__)
import scipy.misc
import time
import timeit
import argparse
print('importing modules complete')
os.chdir('/Users/justinchan/dropbox/thesis/python')

# dependant functions
def importfiles():
    #import test videos
    tv1 = cv2.VideoCapture('1.mp4')
    tv2 = cv2.VideoCapture('2.mp4')
    tv3 = cv2.VideoCapture('3.mp4')
    tv4 = cv2.VideoCapture('4.mp4')
    tv5 = cv2.VideoCapture('5.mp4')
    print('importing videos complete')

    #import test photos
    tp1 = cv2.imread('5.jpg',cv2.IMREAD_UNCHANGED)
    tp2 = cv2.imread('6.jpg',cv2.IMREAD_UNCHANGED)
    print('importing photos complete') 

    tv = [tv1,tv2,tv3,tv4,tv5]
    tp = [tp1,tp2]
    return[tv,tp]  
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
def killkinect(ctx,dev):
    freenect.close_device(dev)
    freenect.shutdown(ctx)
def kill():
    [ctx,dev] = initkinect()
    cv2.destroyAllWindows()
    freenect.sync_stop()
    freenect.stop_video(dev)
    freenect.stop_depth(dev)
    freenect.close_device(dev)
    freenect.shutdown(ctx)
    quit()

# testing functions
def testkinect():
    # test leds and tilts
    [ctx,dev] = initkinect()
    for i in range(1,6):
        freenect.set_led(dev,i)
        cv2.waitKey(2000)
    freenect.set_tilt_degs(dev,0)
    print('complete led')
    cv2.waitKey(3000)
    #test tilt
    freenect.set_tilt_degs(dev,-50)
    cv2.waitKey(3000)
    freenect.set_tilt_degs(dev,50)
    cv2.waitKey(3000)
    freenect.set_tilt_degs(dev,0)
    cv2.waitKey(1)
    print('complete tilt')
    freenect.shutdown(ctx)
def testkinect1():
    # test video modes
    [ctx,dev] = initkinect()
    print(freenect.get_video_format(dev))
    freenect.set_video_mode(dev,1,1)
    print(freenect.get_video_format(dev))
    killkinect(ctx,dev)
    old_time = time.time()
    while time.time()-old_time < 10:
        cv2.imshow('Video', get_video1())
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1000)
    freenect.sync_stop()

def testmovewindow():
    print ('testing move window')
    importfiles()
    cv2.namedWindow('testing',cv2.WINDOW_NORMAL)
    cv2.imshow('testing',tp2)
    cv2.waitKey(3000)
    cv2.resizeWindow('testing',100,100)
    cv2.moveWindow('testing',50,0);
    cv2.waitKey(3000)
    cv2.moveWindow('testing',100,10);
    cv2.resizeWindow('testing',200,200)
    cv2.waitKey(3000)
    cv2.moveWindow('testing',200,20);
    cv2.resizeWindow('testing',300,300)
    cv2.waitKey(3000)
    print('5')
    cv2.destroyAllWindows()
    print('complete')
def displayimage(img):
    print('test display image');
    cv2.namedWindow('test',cv2.WINDOW_NORMAL);
    cv2.waitKey(1)
    cv2.imshow('test',img);
    cv2.waitKey(1000);
    cv2.destroyAllWindows();
    cv2.waitKey(1);
    print('complete');
def playvid(vid):
    print('test play vid')
    while vid.isOpened():
        ret,frame = vid.read()
        if not ret:
            print('oh no')
            break
        cv2.imshow('image',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('q pressed')
            break
    cv2.destroyAllWindows()
    vid.release()
    cv2.waitKey(1);
    print('completed');

def playkinect():
    while 1:
        cv2.imshow('Depth',get_depth())
        cv2.imshow('Video',get_video())
        if cv2.waitKey(10) == 27:
            break
def recordwebcam():
    cap = cv2.VideoCapture(0)
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter()
    success = out.open('output.mp4',code, 20,(1280,720),True)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,180)
            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
def recordkinect():
    # record video for 30 seconds
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vid = cv2.VideoWriter()
    success2 = vid.open('video.mp4',code,20,(640,480),True)
    old_time = time.time()
    while time.time()-old_time < 120:
        vid.write(get_video())
        cv2.waitKey(1)
        print(time.time())
    # depth.release()
    vid.release()
def recordkinect1():
 # record depth for 30 seconds
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    depth = cv2.VideoWriter()
    success = depth.open('depth.mp4',code,20,(640,480),True)
    old_time = time.time()
    while time.time() - old_time < 30:
        dep = get_depth()
        image = cv2.cvtColor(dep,cv2.COLOR_GRAY2BGR)
        depth.write(image)
        cv2.waitKey(10)
    depth.release()
def recordkinect2():
    # record vid and depth side by side
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    dual = cv2.VideoWriter()
    success = dual.open('dual.mp4',code,20,(1280,480),True)
    old_time = time.time()
    while(time.time()-old_time<120):
        dep = get_depth()
        vid = get_video()
        depth = cv2.cvtColor(dep,cv2.COLOR_GRAY2BGR)
        output = np.zeros((480, 640*2, 3), dtype="uint8")
        output[0:480,0:640] = vid
        output[0:480,640:1280] = depth
        # cv2.imshow('lol',output)
        dual.write(output)
    dual.release()

def basic(vid):
    print('beginning basic')
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    # vidout = cv2.VideoWriter()
    # success2 = vidout.open('basic.mp4',code,20,(640,480),True)
    start_time = time.time()
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('bg',bgFrame)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=30000)):
        ret,vidFrame = vid.read();
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY);
        bsFrame = cv2.absdiff(vidFrame,bgFrame);
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        cv2.imshow('vid',vidFrame)
        cv2.imshow('thr',thrFrame)
        # vidout.write(bsFrame)
        cv2.waitKey(1)
    end_time = time.time();
    print('process time =',(end_time-start_time))
    vid.release();
    # vidout.release();
    cv2.destroyAllWindows();
    cv2.waitKey(1);
    print('finished basic');
def temporaldifferencing(vid):
    print('beginning temporal differencing')
    start_time = time.time()
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=30000)):
        ret,vidFrame = vid.read();
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY);
        bsFrame = cv2.absdiff(vidFrame,bgFrame);
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        bgFrame = vidFrame
        cv2.imshow('bg',bgFrame)
        cv2.imshow('bs',bsFrame)
        cv2.imshow('thr',thrFrame)
        cv2.waitKey(1)
    end_time = time.time();
    print('process time =',(end_time-start_time))
    vid.release();
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
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=30000)):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(vidFrame,avgFrame,0.01)
        bgFrame = cv2.convertScaleAbs(avgFrame)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        cv2.imshow('bs',bsFrame)
        cv2.imshow('th2',thrFrame)
        cv2.waitKey(1)
    end_time = time.time()
    print(end_time-start_time)   
    cv2.destroyAllWindows()
    vid.release()
def approximatemedian(vid):
    print('starting approximate median')
    # read input video
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY) 
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    # print(rows)
    # print(columns)
    start_time = time.time()
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=60000)):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        for i in range(0,int(width)):
            for j in range(0,int(height)):
                if vidFrame[j,i] > bgFrame[j,i]:
                    bgFrame[j,i] = bgFrame[j,i] + 10
                elif vidFrame[j,i] < bgFrame[j,i]:
                    bgFrame[j,i]= bgFrame[j,i] - 10
        cv2.imshow('th',thrFrame)
        cv2.imshow('vid',vidFrame)
        cv2.imshow('bs',bsFrame)
        cv2.imshow('bg',bgFrame)
        cv2.waitKey(1)
    end_time = time.time()
    print(end_time-start_time)
    cv2.destroyAllWindows()
    vid.release()
def knn(vid):
    fgbg = cv2.createBackgroundSubtractorKNN()
    while 1:
        ret,frame = vid.read()
        fgmask = fgbg.apply(frame)
        bgmask = fgbg.getBackgroundImage()
        cv2.imshow('frame',fgmask)
        cv2.imshow('bg',bgmask)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    vid.release()  
def gmg(vid):
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    while 1:
        ret,frame = vid.read()
        fgmask = fgbg.apply(frame)
        bgmask = fgbg.getBackgroundImage()
        cv2.imshow('frame',fgmask)
        cv2.imshow('bg',bgmask)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    vid.release()
def mog(vid):
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    while 1:
        ret,frame = vid.read()
        fgmask = fgbg.apply(frame)
        # bgmask = fgbg.getBackgroundImage()
        cv2.imshow('frame',fgmask)
        # cv2.imshow('bg',bgmask)
        cv2.waitKey(1)
    vid.release()
    cv2.destroyAllWindows()
def mog2(vid):   
    # requires opencv 2.4
    fgbg = cv2.BackgroundSubtractorMOG2()
    oldtime = vid.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.cv.CV_CAP_PROP_POS_MSEC)-oldtime<=30000)):
        ret,frame = vid.read()
        fgmask = fgbg.apply(frame)
        cv2.imshow('frame',fgmask)
        cv2.waitKey(1)
    vid.release()
    cv2.destroyAllWindows()