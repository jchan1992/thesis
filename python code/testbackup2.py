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

# dependant functions
def importfiles():
    #import test videos
    tv1 = cv2.VideoCapture('1.mp4')
    tv2 = cv2.VideoCapture('2.mp4')
    tv3 = cv2.VideoCapture('3.mp4')
    tv4 = cv2.VideoCapture('4.mp4')
    tv5 = cv2.VideoCapture('5.mp4')
    
    tx1 = cv2.VideoCapture('dual.mp4')
    tx2 = cv2.VideoCapture('newvid.mp4')
    tx3 = cv2.VideoCapture('newvid1.mp4')
    print('importing videos complete')
    #import test photos
    tp1 = cv2.imread('5.jpg',cv2.IMREAD_UNCHANGED)
    tp2 = cv2.imread('6.jpg',cv2.IMREAD_UNCHANGED)
    print('importing photos complete') 
    # return video and photo arrays
    tv = [tv1,tv2,tv3,tv4,tv5]
    tp = [tp1,tp2]
    tx = [tx1,tx2,tx3]
    return[tv,tp,tx]  
def drawhist(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    MAX = max(hist)
    plot = np.zeros((512,1024))
    for i in range(255):
        x1 = 4*i
        x2 = 4*(i+1)
        y1 = hist[i]*512/MAX
        y2 = hist[i+1]*512/MAX
        cv2.line(plot, (x1,y1), (x2,y2), 1, 3)
    # cv2.imshow("-gray", gray)
    cv2.imshow("-hist", plot)
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
        if cv2.waitKey(1) & 0xFF ==27:
            print('esc pressed')
            break
    cv2.destroyAllWindows()
    vid.release()
    cv2.waitKey(1);
    print('completed');
def stitchim():
    a = cv2.imread('a.png')
    b = cv2.imread('b.png')
    c = np.zeros((480,1280,3), dtype="uint8")
    c[0:480,0:640] = a
    c[0:480,640:1280] = b
    cv2.imwrite('c.png',c)

def playkinect():
    print('display kinect stream')
    while 1:
        cv2.imshow('Depth',get_depth())
        cv2.imshow('Video',get_video())
        if cv2.waitKey(10) == 27:
            break
def playkinect1():
    print('display kinect stream')
    while 1:
        vid = get_video()
        dep = get_depth()

        cv2.imshow('Depth',dep)
        cv2.imshow('Video',vid)
        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) == ord('a'):
            cv2.imwrite('a.png',vid)
        if cv2.waitKey(1) == ord('b'):
            cv2.imwrite('b.png',vid)
def recordwebcam():
    cap = cv2.VideoCapture(0)
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter()
    success = out.open('output.mp4',code, 20,(1280,720),True)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,180)
            print(type(frame))
            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == 27:
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
    while time.time()-old_time < 30:
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
    success = dual.open('dual3.mp4',code,20,(1280,480),True)
    old_time = time.time()
    while(time.time()-old_time<240):
    # while 1:
        dep = get_depth()
        vid = get_video()
        depth = cv2.cvtColor(dep,cv2.COLOR_GRAY2BGR)
        output = np.zeros((480, 640*2, 3), dtype="uint8")
        output[0:480,0:640] = vid
        output[0:480,640:1280] = depth
        # cv2.imshow('lol',output)
        dual.write(output)
    dual.release()
def recordvid(vid):
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    newvid = cv2.VideoWriter()
    success = newvid.open('newvid1.mp4',code,20,(640,480),True)
    while vid.isOpened():
        ret,frame = vid.read()
        if not ret:
            print('oh no')
            break
        nframe = frame[0:480,640:1280]
        # nframe = cv2.cvtColor(nframe,cv2.COLOR_BGR2GRAY)
        # cv2.imshow('new',nframe)
        newvid.write(nframe)
        # cv2.waitKey(1)
    vid.release()
    newvid.release()
def testthresh(vid):
    print('test thresh')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask = np.zeros((height+2, width+2), np.uint8)
    block = 10
    # get first frame
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    # kbgFrame = bgFrame

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Z = kbgFrame.reshape((-1,3))
    # Z = np.float32(Z)
    # K = 2
    # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((kbgFrame.shape))

    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=30000)):
        ret,vidFrame = vid.read();
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY);
        bsFrame = cv2.absdiff(vidFrame,bgFrame);
        
        ret,thrOtsu = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        ret,thrTriangle = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        thrmeanc = cv2.adaptiveThreshold(bsFrame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        thrgauss = cv2.adaptiveThreshold(bsFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
        cv2.floodFill(thrmeanc,mask,(0,0),255);
        cv2.floodFill(thrgauss,mask,(0,0),255)

        Z = bsFrame.reshape((-1,3))
        Z = np.float32(Z)
        K = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((bsFrame.shape))
        ret,thrkmeans = cv2.threshold(res2,0,255,cv2.THRESH_OTSU)

        thrniblack = niblack(bsFrame,10,0.4)
        thrwellner = wellner(bsFrame,10)
        cv2.imshow('niblack',thrniblack)
        cv2.imshow('wellner',thrwellner)
        cv2.imshow('kmeans',thrkmeans)
        cv2.imshow('otsu',thrOtsu)
        cv2.imshow('triangle',thrTriangle)
        cv2.imshow('meanc',thrmeanc)
        cv2.imshow('gauss',thrgauss)
        cv2.waitKey(1)

    cv2.destroyAllWindows();
    cv2.waitKey(1);

    print('finished basic');

# local thresholding techniques
def niblack(im,size,k):
    retim = im.copy()
    cv2.waitKey(10000)
    h,w = retim.shape[:2]
    for i in xrange(0,h,size):
        for j in xrange(0,w,size):
            testimage = retim[i:i+size,j:j+size]
            [mean,stddev] = cv2.meanStdDev(testimage)
            thresh = mean + k*stddev
            ret,retim[i:i+size,j:j+size] = cv2.threshold(retim[i:i+size,j:j+size],thresh,255,cv2.THRESH_BINARY)
    return retim
def sauvola(image,size,k,R):
    retim1 = image.copy()
    h,w = retim1.shape[:2]
    for i in xrange(0,h,size):
        for j in xrange(0,w,size):
            testimage = retim1[i:i+size,j:j+size]
            [mean,stddev] = cv2.meanStdDev(testimage)
            stdr = float(stddev/R)
            thresh = mean + mean*k*stdr - k*mean
            ret,retim1[i:i+size,j:j+size] = cv2.threshold(retim1[i:i+size,j:j+size],thresh,255,cv2.THRESH_BINARY)
    return retim1
def wellner(image,size):
    retim2 = image.copy()
    h,w = retim2.shape[:2]
    for i in xrange(0,h,size):
        for j in xrange(0,w,size):
            ret,retim2[i:i+size,j:j+size] = cv2.threshold(retim2[i:i+size,j:j+size],0,255,cv2.THRESH_OTSU)
    return retim2

# background subtraction algorithms
def basic(vid):
    print('beginning basic')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # set up video writer
    # code = cv2.VideoWriter_fourcc('m','p','4','v')
    # vidout = cv2.VideoWriter()
    # success2 = vidout.open('basic1.mp4',code,fps,(width,height),True)
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
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        # cv2.imshow('vid',vidFrame)
        cv2.imshow('thr',thrFrame)
        thrFrame = cv2.cvtColor(thrFrame,cv2.COLOR_GRAY2BGR)
        # vidout.write(thrFrame)
        cv2.waitKey(1)
    end_time = time.time();
    # print('process time =',(end_time-start_time))
    timet = end_time - start_time
    f = open('results.txt','a')
    f.write('basic process time = %d' % timet)
    f.close()
    vid.release();
    # vidout.release();
    cv2.destroyAllWindows();
    cv2.waitKey(1);
    print('finished basic');
def temporaldifferencing(vid):
    print('beginning temporal differencing')
    # get video properties
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # set up video writer
    # code = cv2.VideoWriter_fourcc('m','p','4','v')
    # vidout = cv2.VideoWriter()
    # success2 = vidout.open('temporal.mp4',code,fps,(width,height),True)
    start_time = time.time()
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,vidFrame = vid.read();
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY);
        bsFrame = cv2.absdiff(vidFrame,bgFrame);
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        thrFrame = cv2.cvtColor(thrFrame,cv2.COLOR_GRAY2BGR)
        bgFrame = vidFrame
        # cv2.imshow('bg',bgFrame)
        # cv2.imshow('bs',bsFrame)
        cv2.imshow('thr',thrFrame)
        # vidout.write(thrFrame)
        cv2.waitKey(1)
    end_time = time.time();
    timet = end_time - start_time
    # print('process time =',(end_time-start_time))
    f = open('results.txt','a')
    f.write('temporal process time = %d//' % timet)
    f.close()
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
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=10000)):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(vidFrame,avgFrame,0.5)
        bgFrame = cv2.convertScaleAbs(avgFrame)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        cv2.imshow('vid',vidFrame)
        cv2.imshow('bg',bgFrame)
        cv2.imshow('bs',bsFrame)
        cv2.imshow('th',thrFrame)
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
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    # print(rows)
    # print(columns)
    start_time = time.time()
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=10000)):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        for i in xrange(height):
            for j in xrange(width):
                k = vidFrame.item(i,j)
                l = bgFrame.item(i,j) 
                if k>l:
                    bgFrame.itemset((i,j),l+10)
                elif k<l:
                    bgFrame.itemset((i,j),l-10)
        # cv2.imshow('th',thrFrame)
        # cv2.imshow('vid',vidFrame)
        cv2.imshow('bs',bsFrame)
        # cv2.imshow('bg',bgFrame)
        cv2.waitKey(1)
    end_time = time.time()
    print(end_time-start_time)
    cv2.destroyAllWindows()
    vid.release()
def approximatemedian1(vid):
    print('starting approximate median')
    # read input video
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY) 
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    length = height*width
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    # print(rows)
    # print(columns)
    start_time = time.time()
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=10000)):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        for i in xrange(length):
                k = vidFrame.item(i)
                l = bgFrame.item(i) 
                if k>l:
                    bgFrame.itemset((i),l+10)
                elif k<l:
                    bgFrame.itemset((i),l-10)
        # cv2.imshow('th',thrFrame)
        # cv2.imshow('vid',vidFrame)
        cv2.imshow('bs',bsFrame)
        # cv2.imshow('bg',bgFrame)
        cv2.waitKey(1)
    end_time = time.time()
    print(end_time-start_time)
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
# def knn(vid):
#     fgbg = cv2.createBackgroundSubtractorKNN()
#     while 1:
#         ret,frame = vid.read()
#         fgmask = fgbg.apply(frame)
#         bgmask = fgbg.getBackgroundImage()
#         cv2.imshow('frame',fgmask)
#         cv2.imshow('bg',bgmask)
#         cv2.waitKey(1)
#     cv2.destroyAllWindows()
#     vid.release()  