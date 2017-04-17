
# legacy functions probably doesnt work without putting it back into test.py

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

    tv1 = cv2.VideoCapture('1.mp4')
    tv2 = cv2.VideoCapture('2.mp4')
    tv3 = cv2.VideoCapture('3.mp4')
    tv4 = cv2.VideoCapture('4.mp4')
    tv5 = cv2.VideoCapture('5.mp4')
    
       #import test photos
    tp1 = cv2.imread('5.jpg',cv2.IMREAD_UNCHANGED)
    tp2 = cv2.imread('6.jpg',cv2.IMREAD_UNCHANGED)
    print('importing photos complete') 

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
def gmg(vid):
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,frame = vid.read()
        fgmask = fgbg.apply(frame)
        # bgmask = fgbg.getBackgroundImage()
        cv2.imshow('frame',fgmask)
        # cv2.imshow('bg',bgmask)
        cv2.waitKey(1)
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
    
def mybs(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    ret,bgframe = vid.read()
    videobg = bgframe[0:height,0:width/2]
    depthbgo = bgframe[0:height,width/2:width]
    depthbgo = cv2.cvtColor(depthbgo,cv2.COLOR_BGR2GRAY)
    depthbgp = depthbgo.copy()
    while(1):
        ret,frame = vid.read()
        video = frame[0:height,0:width/2]
        depth = frame[0:height,width/2:width]
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        
        # vidmask = fgbg.apply(video)

        depthmask = cv2.absdiff(depth,depthbgo)
        # depthmask1 = cv2.subtract(depth,depthbgp)
        # depthbgp = depth

        ret,depthmask = cv2.threshold(depthmask,0,255,cv2.THRESH_TRIANGLE)
        # ret,depthmask1 = cv2.threshold(depthmask1,0,255,cv2.THRESH_TRIANGLE)
        
        # depthmask2 = cv2.bitwise_and(depthmask,depthmask1)
        # depthmask3 = cv2.absdiff(depth,depthbgo)

        # ret,depthmask2 = cv2.threshold(depthmask2,0,255,cv2.THRESH_TRIANGLE)
        # ret,depthmask3 = cv2.threshold(depthmask3,0,255,cv2.THRESH_TRIANGLE) 

        fmask = cv2.bitwise_or(vidmask,depthmask)
        cv2.imshow('f',fmask)
        cv2.imshow('vid',vidmask)
        cv2.imshow('dep',depthmask)
        # cv2.imshow('dm1',depthmask1)
        # cv2.imshow('dm',depthmask)
        # cv2.imshow('dm2',depthmask2)
        # cv2.imshow('dm3',depthmask3)
        cv2.waitKey(1)
        if cv2.waitKey(1) &0xff == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(10)
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
        nframe = frame[0:480,0:640]
        # nframe = cv2.cvtColor(nframe,cv2.COLOR_BGR2GRAY)
        # cv2.imshow('new',nframe)
        newvid.write(nframe)
        # cv2.waitKey(1)
    vid.release()
    newvid.release()
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
def getbsframe1(vid):
    ret,bgFrame = vid.read()
    imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    i = 0
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    kbgFrame = bgFrame
    Z = kbgFrame.reshape((-1,3))
    Z = np.float32(Z)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res1= res.reshape((kbgFrame.shape))
    while(imtime<30000):
        ret,vidFrame = vid.read()
        kvidFrame = vidFrame
        Z = kvidFrame.reshape((-1,3))
        Z = np.float32(Z)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((kvidFrame.shape))
        
        bsFrame = cv2.absdiff(res1,res2)
        bsFrame = cv2.cvtColor(bsFrame,cv2.COLOR_BGR2GRAY)
        ret,bsFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        print(ret)
        cv2.imwrite('%03dkmeanso.jpg' % i,bsFrame)

        bgFrame = res2
        i += 1
        imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
def getbsframe2(vid):
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    i = 0
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    while(imtime<30000):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        
        kbsFrame = bsFrame
        Z = kbsFrame.reshape((-1,3))
        Z = np.float32(Z)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((kbsFrame.shape))
        
        ret,res2 = cv2.threshold(res2,0,255,cv2.THRESH_TRIANGLE)
        print ret

        cv2.imwrite('%03dkmeans.jpg' % i,res2)

        bgFrame = vidFrame
        i += 1
        imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
def getvid(vid):
    ret,bgFrame = vid.read()
    imtime = vid.get(cv2.CAP_PROP_POS_MSEC) 
    i = 0
    cv2.imwrite('%03dvid.jpg' % i,bgFrame)
    while(imtime<30000):
        ret,vidFrame = vid.read()
        cv2.imwrite('%03dvid.jpg' % i,vidFrame)
        i += 1
        imtime = vid.get(cv2.CAP_PROP_POS_MSEC)   
def getthreshframe(vid):
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    i = 0
    while(imtime<30000):
        ret,vidFrame = vid.read()
        vidFrame =cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        ret,thrOtsu = cv2.threshold(bsFrame,0,255,cv2.THRESH_OTSU)
        ret,thrTriangle = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        thrniblack10 = niblack(bsFrame,10,0.4)
        thrsauv10 = sauvola(bsFrame,10,0.4,128)
        thrwellner10 = wellner(bsFrame,10)
        thrniblack100 = niblack(bsFrame,100,0.4)
        thrsauv100 = sauvola(bsFrame,100,0.4,128)
        thrwellner100 = wellner(bsFrame,100)

        cv2.imwrite('%03dotsu.jpg' % i,thrOtsu)
        cv2.imwrite('%03dtriangle.jpg' % i,thrTriangle)
        cv2.imwrite('%03dniblack10.jpg' % i,thrniblack10)
        cv2.imwrite('%03dniblack100.jpg' % i,thrniblack100)
        cv2.imwrite('%03dsauv10.jpg' % i,thrsauv10)
        cv2.imwrite('%03dsauv100.jpg' % i,thrsauv100)
        cv2.imwrite('%03dwellner10.jpg' % i,thrwellner10)
        cv2.imwrite('%03dwellner100.jpg' % i,thrwellner100)

        i += 1
        bgFrame = vidFrame
        imtime = vid.get(cv2.CAP_PROP_POS_MSEC) 

def stitchim(num):
    bs = cv2.imread('result thresholding/%03dbs.jpg' % num)
    kmeans = cv2.imread('result thresholding/%03dkmeans.jpg' % num)
    kmeanso = cv2.imread('result thresholding/%03dkmeanso.jpg' % num)
    niblack10 = cv2.imread('result thresholding/%03dniblack10.jpg' % num)
    niblack100 = cv2.imread('result thresholding/%03dniblack100.jpg' % num)
    otsu = cv2.imread('result thresholding/%03dotsu.jpg' % num)
    sauv10 = cv2.imread('result thresholding/%03dsauv10.jpg' % num)
    sauv100 = cv2.imread('result thresholding/%03dsauv100.jpg' % num)
    triangle = cv2.imread('result thresholding/%03dtriangle.jpg' % num)
    video = cv2.imread('result thresholding/%03dvid.jpg' % num)
    # gt = cv2.imread('result thresholding/%03dgt.jpg' % num)
    wellner10 = cv2.imread('result thresholding/%03dwellner10.jpg' % num)
    wellner100 = cv2.imread('result thresholding/%03dwellner100.jpg' % num)
    
    cv2.putText(bs,"Background Subtraction",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(video,"Ground Truth",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(kmeans,"Kmeans",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(kmeanso,"Kmeans",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(otsu,"Otsu",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(triangle,"Triangle",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(niblack10,"Niblack blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(niblack100,"Niblack blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(sauv10,"Sauvola blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(sauv100,"Sauvola blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(wellner10,"Wellner blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(wellner100,"Wellner blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    

    height,width = bs.shape[:2]
    newim = np.zeros((height*6, width*2, 3), dtype="uint8")
    newim[0:height,0:width] = bs
    newim[0:height,width:2*width] = video

    newim[height:2*height,0:width] = otsu
    newim[height:2*height,width:2*width] = triangle
    
    newim[2*height:3*height,0:width] = kmeans
    newim[2*height:3*height,width:2*width] = kmeanso
    
    newim[3*height:4*height,0:width] = niblack10
    newim[3*height:4*height,width:2*width] = niblack100
    
    newim[4*height:5*height,0:width] = sauv10
    newim[4*height:5*height,width:2*width] = sauv100
    
    newim[5*height:6*height,0:width] = wellner10
    newim[5*height:6*height,width:2*width] = wellner100

    newim[0:6*height,width-1:width+1,:] = (0,255,0)
    newim[0:6*height,0:2,:] = (0,255,0)
    newim[0:6*height,2*width-2:2*width,:] = (0,255,0)
    newim[0:5,0:2*width,:] = (0,255,0)
    newim[height-1:height+1,0:2*width,:] = (0,255,0)
    newim[2*height-1:2*height+1,0:2*width,:] = (0,255,0)
    newim[3*height-1:3*height+1,0:2*width,:] = (0,255,0)
    newim[4*height-1:4*height+1,0:2*width,:] = (0,255,0)
    newim[5*height-1:5*height+1,0:2*width,:] = (0,255,0)
    newim[6*height-1:6*height+1,0:2*width,:] = (0,255,0)        
    cv2.imwrite('result thresholding/%03dstitch.jpg' % num,newim)    
def stitchim1(num):
    bs = cv2.imread('result thresholding/%03dbs.jpg' % num)
    kmeans = cv2.imread('result thresholding/%03dkmeans.jpg' % num)
    kmeanso = cv2.imread('result thresholding/%03dkmeanso.jpg' % num)
    niblack10 = cv2.imread('result thresholding/%03dniblack10.jpg' % num)
    niblack100 = cv2.imread('result thresholding/%03dniblack100.jpg' % num)
    otsu = cv2.imread('result thresholding/%03dotsu.jpg' % num)
    sauv10 = cv2.imread('result thresholding/%03dsauv10.jpg' % num)
    sauv100 = cv2.imread('result thresholding/%03dsauv100.jpg' % num)
    triangle = cv2.imread('result thresholding/%03dtriangle.jpg' % num)
    video = cv2.imread('result thresholding/%03dvid.jpg' % num)
    # gt = cv2.imread('result thresholding/%03dgt.jpg' % num)
    wellner10 = cv2.imread('result thresholding/%03dwellner10.jpg' % num)
    wellner100 = cv2.imread('result thresholding/%03dwellner100.jpg' % num)
    
    cv2.putText(bs,"Background Subtraction",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(video,"Ground Truth",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(kmeans,"Kmeans",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(kmeanso,"Kmeans",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(otsu,"Otsu",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(triangle,"Triangle",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(niblack10,"Niblack blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(niblack100,"Niblack blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(sauv10,"Sauvola blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(sauv100,"Sauvola blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(wellner10,"Wellner blocksize = 10",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(wellner100,"Wellner blocksize = 100",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    

    height,width = bs.shape[:2]
    newim = np.zeros((height*3, width*4, 3), dtype="uint8")
    newim[0:height,0:width] = bs
    newim[0:height,width:2*width] = video
    newim[0:height,2*width:3*width] = otsu
    newim[0:height,3*width:4*width] = triangle
    
    newim[1*height:2*height,0:width] = kmeans
    newim[1*height:2*height,width:2*width] = kmeanso
    newim[1*height:2*height,2*width:3*width] = niblack10
    newim[1*height:2*height,3*width:4*width] = niblack100
    
    newim[2*height:3*height,0:width] = sauv10
    newim[2*height:3*height,width:2*width] = sauv100
    newim[2*height:3*height,2*width:3*width] = wellner10
    newim[2*height:3*height,3*width:4*width] = wellner100

    newim[0:3*height,0:2,:] = (0,255,0)
    newim[0:3*height,4*width-2:4*width,:] = (0,255,0)
    newim[0:3*height,width-1:width+1,:] = (0,255,0)
    newim[0:3*height,2*width-1:2*width+1,:] = (0,255,0)
    newim[0:3*height,3*width-1:3*width+1,:] = (0,255,0)
    
    newim[0:2,0:width*3,:] = (0,255,0)
    newim[height-1:height+1,0:width*3,:] = (0,255,0)
    newim[2*height-1:2*height+1,0:width*3,:] = (0,255,0)
    newim[3*height-1:3*height+1,0:width*3,:] = (0,255,0)
    
    cv2.imwrite('result thresholding/%03dstitch1.jpg' % num,newim) 

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
    kbgFrame = bgFrame.copy()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    Z = kbgFrame.reshape((-1,3))
    Z = np.float32(Z)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((kbgFrame.shape))

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
def testbsframe1(vid):
    f = open('thresspeed.txt','a')
    ret,bgFrame = vid.read()
    imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    i = 0
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    kbgFrame = bgFrame
    Z = kbgFrame.reshape((-1,3))
    Z = np.float32(Z)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res1= res.reshape((kbgFrame.shape))
    while(imtime<10000):
        ret,vidFrame = vid.read()
        kvidFrame = vidFrame
        start = int(round(time.time()*1000))
        Z = kvidFrame.reshape((-1,3))
        Z = np.float32(Z)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((kvidFrame.shape))
        
        bsFrame = cv2.absdiff(res1,res2)
        bsFrame = cv2.cvtColor(bsFrame,cv2.COLOR_BGR2GRAY)
        ret,bsFrame = cv2.threshold(bsFrame,0,255,cv2.THRESH_TRIANGLE)
        end = int(round(time.time()*1000))
        final = end - start
        f.write('%d \t %d \t kmeans double process time = %d\n' % (end,start,final))
        bgFrame = res2
        i += 1
        imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
def testbsframe2(vid):
    f = open('thresspeed.txt','a')
    ret,bgFrame = vid.read()
    bgFrame = cv2.cvtColor(bgFrame,cv2.COLOR_BGR2GRAY)
    imtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    i = 0
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    while(imtime<10000):
        ret,vidFrame = vid.read()
        vidFrame = cv2.cvtColor(vidFrame,cv2.COLOR_BGR2GRAY)
        bsFrame = cv2.absdiff(vidFrame,bgFrame)
        
        kbsFrame = bsFrame
        start = int(round(time.time()*1000))
        Z = kbsFrame.reshape((-1,3))
        Z = np.float32(Z)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((kbsFrame.shape))
        
        ret,res2 = cv2.threshold(res2,0,255,cv2.THRESH_TRIANGLE)
        end = int(round(time.time()*1000))
        final = end - start
        f.write('%d \t %d \t kmeans bs process time = %d\n' % (end,start,final))
        bgFrame = vidFrame
        i += 1
        imtime = vid.get(cv2.CAP_PROP_POS_MSEC)   

# local thresholding techniques
def niblack(im,size,k):
    retim = im.copy()
    # cv2.waitKey(10000)
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
def niblackg(im,size,k):
    retim = im.copy()
    # cv2.waitKey(10000)
    [mean,stddev] = cv2.meanStdDev(retim)
    thresh = mean + k*stddev
    ret,retim = cv2.threshold(retim,thresh,255,cv2.THRESH_BINARY)
    return retim
def sauvolag(image,size,k,R):
    retim1 = image.copy()
    testimage = retim1[i:i+size,j:j+size]
    [mean,stddev] = cv2.meanStdDev(retim)
    stdr = float(stddev/R)
    thresh = mean + mean*k*stdr - k*mean
    ret,retim1 = cv2.threshold(retim1,thresh,255,cv2.THRESH_BINARY)
    return retim1

def mybs1(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    # fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
    ret,bgframe = vid.read()
    videobg = bgframe[0:height,0:width/2]
    videobgo = cv2.cvtColor(videobg,cv2.COLOR_BGR2GRAY)
    videobgo = cv2.equalizeHist(videobgo)
    videobgp = videobgo.copy()
    depthbg = bgframe[0:height,width/2:width]
    depthbgo = cv2.cvtColor(depthbg,cv2.COLOR_BGR2GRAY)
    # depthbgo = cv2.equalizeHist(depthbgo)
    depthbgp = depthbgo.copy()
    while(1):
        ret,frame = vid.read()
        video = frame[0:height,0:width/2]
        depth = frame[0:height,width/2:width]
        video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        video = cv2.equalizeHist(video)
        # depth = cv2.equalizeHist(depth)

        ret,frame1 = vid.read()
        video1 = frame[0:height,0:width/2]
        depth1 = frame[0:height,width/2:width]
        video1 = cv2.cvtColor(video1,cv2.COLOR_BGR2GRAY)
        depth1 = cv2.cvtColor(depth1,cv2.COLOR_BGR2GRAY)
        video1 = cv2.equalizeHist(video1)
        # depth1 = cv2.equalizeHist(depth1)

        # vidmask = fgbg.apply(video)
        depbasic = cv2.absdiff(depth,depthbgo)
        deptemporal = cv2.absdiff(depth,depthbgp)
        ret,depbasic = cv2.threshold(depbasic,0,255,cv2.THRESH_TRIANGLE)
        ret,deptemporal = cv2.threshold(deptemporal,0,255,cv2.THRESH_TRIANGLE)
        
        vidbasic = cv2.absdiff(video,videobgo)
        vidtemporal = cv2.absdiff(video,videobgp)
        ret,vidbasic = cv2.threshold(vidbasic,0,255,cv2.THRESH_TRIANGLE)
        ret,vidtemporal = cv2.threshold(vidtemporal,0,255,cv2.THRESH_TRIANGLE)

        # depmask2 = cv2.subtract(depmask,depmask1)
        # fmask = cv2.bitwise_or(vidmask,depmask)
        # fmask = cv2.bitwise_or(fmask,depmask1)
        cv2.imshow('depbasic',depbasic)
        cv2.imshow('deptemporal',deptemporal)
        cv2.imshow('vidbasic',vidbasic)
        cv2.imshow('vidtemporal',vidtemporal)

        mask1 = cv2.subtract(depbasic,deptemporal)
        mask2 = cv2.subtract(depbasic,vidbasic)
        mask3 = cv2.subtract(depbasic,vidtemporal)

        cv2.imshow('mask1',mask1)
        cv2.imshow('mask2',mask2)
        cv2.imshow('mask3',mask3)

        mask4 = cv2.bitwise_and(depth,depth1)
        mask4 = cv2.bitwise_and(mask4,depthbgp)
        mask41 = cv2.bitwise_or(depth,depth1)
        mask41 = cv2.bitwise_or(mask41,depthbgp)
        mask5 = cv2.bitwise_and(video,video1)
        mask5 = cv2.bitwise_and(mask5,videobgp)
        mask51 = cv2.bitwise_or(video,video1)
        mask51 = cv2.bitwise_or(mask51,videobgp)
        ret,mask4 = cv2.threshold(mask4,0,255,cv2.THRESH_TRIANGLE)
        ret,mask5 = cv2.threshold(mask5,0,255,cv2.THRESH_TRIANGLE)
        ret,mask41 = cv2.threshold(mask41,0,255,cv2.THRESH_TRIANGLE)
        ret,mask51 = cv2.threshold(mask51,0,255,cv2.THRESH_TRIANGLE)
        cv2.imshow('mask4',mask4)
        cv2.imshow('mask41',mask41)
        cv2.imshow('mask5',mask5)
        cv2.imshow('mask51',mask51)

        mask6 = cv2.bitwise_or(depbasic,vidbasic)
        mask6 = cv2.subtract(mask6,vidtemporal)
        mask8 = cv2.subtract(depbasic,mask5)
        mask81 = cv2.subtract(depbasic,mask51)

        mask7 = cv2.subtract(depbasic,mask4)
        mask71 = cv2.subtract(depbasic,mask41)
        mask72 = cv2.bitwise_and(depbasic,mask4)

        mask9 = cv2.bitwise_and(depbasic,vidbasic)
        mask9 = cv2.subtract(mask9,deptemporal)
        cv2.imshow('mask72',mask72)
        cv2.imshow('mask7',mask7)
        cv2.imshow('mask6',mask6)
        cv2.imshow('mask8',mask8)
        cv2.imshow('mask71',mask71)
        cv2.imshow('mask81',mask81)
        cv2.imshow('mask9',mask9)
        if cv2.waitKey(1) == 27:
            break

        videobgp = video
        depthbgp = depth        
def mybs2(vid):
    # tried depth mog doesnt work 
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
    ret,bgframe = vid.read()
    videobg = bgframe[0:height,0:width/2]
    depthbgo = bgframe[0:height,width/2:width]
    depthbgo = cv2.cvtColor(depthbgo,cv2.COLOR_BGR2GRAY)
    depthbgp = depthbgo.copy()
    while(1):
        ret,frame = vid.read()
        video = frame[0:height,0:width/2]
        depth = frame[0:height,width/2:width]
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        
        vidmask = fgbg.apply(video)
        
        depmask = cv2.absdiff(depth,depthbgo)

        depthmog = depmask.copy()
        # ret,depthmog = cv2.threshold(depthmog,0,255,cv2.THRESH_TRIANGLE)
        depthmog = cv2.cvtColor(depthmog,cv2.COLOR_GRAY2BGR)
        depmask0 = fgbg.apply(depthmog)
        # depmask1 = cv2.absdiff(depth,depthbgp)
        
        ret,depmask = cv2.threshold(depmask,0,255,cv2.THRESH_TRIANGLE)
        # ret,depmask1 = cv2.threshold(depmask1,0,255,cv2.THRESH_TRIANGLE)
        # depmask2 = cv2.subtract(depmask,depmask1)
        # fmask = cv2.bitwise_or(vidmask,depmask)
        # fmask = cv2.bitwise_or(fmask,depmask1)
        fmask = cv2.bitwise_and(depmask0,depmask)

        cv2.imshow('dep0',depmask0)
        cv2.imshow('dep',depmask)
        # cv2.imshow('dep1',depmask1)
        # cv2.imshow('dep2',depmask2)        
        cv2.imshow('final',fmask)

        cv2.waitKey(1)
        depthbgp = depth
def mybs3(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    ret,bgframe = vid.read()
    videobg = bgframe[0:height,0:width/2]
    video0 = cv2.cvtColor(videobg,cv2.COLOR_BGR2GRAY)
    video0 = cv2.equalizeHist(video0)
    # videobgp = videobgo.copy()
    depthbg = bgframe[0:height,width/2:width]
    depth0 = cv2.cvtColor(depthbg,cv2.COLOR_BGR2GRAY)
    # depthbgo = cv2.equalizeHist(depthbgo)
    # depthbg0 = depthbgo.copy()

    while(1):
        ret,frame = vid.read()
        video = frame[0:height,0:width/2]
        depth = frame[0:height,width/2:width]
        video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        video = cv2.equalizeHist(video)

        ret,frame1 = vid.read()
        video1 = frame1[0:height,0:width/2]
        depth1 = frame1[0:height,width/2:width]
        video1 = cv2.cvtColor(video1,cv2.COLOR_BGR2GRAY)
        depth1 = cv2.cvtColor(depth1,cv2.COLOR_BGR2GRAY)
        video1 = cv2.equalizeHist(video1)

        vidmask1 = cv2.absdiff(video,video0)
        vidmask2 = cv2.absdiff(video,video1)
        ret,vidmask1 = cv2.threshold(vidmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,vidmask2 = cv2.threshold(vidmask2,ret,255,cv2.THRESH_BINARY)
        vidmaskf = cv2.bitwise_and(vidmask1,vidmask2)

        # depmask1 = cv2.absdiff(depth,depth0)
        # depmask2 = cv2.absdiff(depth,depth1)
        # ret,depmask1 = cv2.threshold(depmask1,0,255,cv2.THRESH_TRIANGLE)
        # ret,depmask2 = cv2.threshold(depmask2,ret,255,cv2.THRESH_BINARY)
        # depmaskf = cv2.bitwise_and(depmask1,depmask2)

        # fmask1 = cv2.bitwise_and(depmaskf,vidmaskf)
        # fmask2 = cv2.bitwise_or(depmaskf,vidmaskf)
        # cv2.imshow('dep1',depmask1)
        cv2.imshow('vid1',vidmask1)
        # cv2.imshow('dep',depmaskf)
        # cv2.imshow('vid',vidmaskf)
        # cv2.imshow('1',fmask1)
        # cv2.imshow('2',fmask2)
        cv2.waitKey(1)
        # if cv2.waitKey(1) == 27:
        #     break
        video0 = video
        # depth0 = depth 
def mybs4(vid):
    # interim final design
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # set up video writer
    code = cv2.VideoWriter_fourcc('m','p','4','v')
    vidout = cv2.VideoWriter()
    success2 = vidout.open('depthresults.mp4',code,fps,(width/2,height),True)
    ret,bgframe = vid.read()
    depthbgo = bgframe[0:height,width/2:width]
    depthbgo = cv2.cvtColor(depthbgo,cv2.COLOR_BGR2GRAY)
    oldtime = vid.get(cv2.CAP_PROP_POS_MSEC)
    while(vid.isOpened() & (vid.get(cv2.CAP_PROP_POS_MSEC)-oldtime<=180000)):
        ret,frame = vid.read()
        depth = frame[0:height,width/2:width]
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        depthmask = cv2.absdiff(depth,depthbgo)
        ret,depthmask = cv2.threshold(depthmask,0,255,cv2.THRESH_TRIANGLE)
        depthmask = cv2.cvtColor(depthmask,cv2.COLOR_GRAY2BGR)
        # vidout.write(depthmask)
    cv2.destroyAllWindows()
    cv2.waitKey(10)     
def mybs5(vid):
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    ret,bgframe = vid.read()
    videobg = bgframe[0:height,0:width/2]
    videobg = cv2.cvtColor(videobg,cv2.COLOR_BGR2GRAY)
    videobg = cv2.equalizeHist(videobg)
    video0 = videobg.copy()
    # videobgp = videobgo.copy()
    depthbg = bgframe[0:height,width/2:width]
    depthbg = cv2.cvtColor(depthbg,cv2.COLOR_BGR2GRAY)
    # depthbgo = cv2.equalizeHist(depthbgo)
    depth0 = depthbg.copy()

    while(1):
        ret,frame = vid.read()
        video = frame[0:height,0:width/2]
        depth = frame[0:height,width/2:width]
        video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        video = cv2.equalizeHist(video)

        ret,frame1 = vid.read()
        video1 = frame1[0:height,0:width/2]
        depth1 = frame1[0:height,width/2:width]
        video1 = cv2.cvtColor(video1,cv2.COLOR_BGR2GRAY)
        depth1 = cv2.cvtColor(depth1,cv2.COLOR_BGR2GRAY)
        video1 = cv2.equalizeHist(video1)

        vidmask1 = cv2.absdiff(video,video0)
        vidmask2 = cv2.absdiff(video,video1)
        vidmask3 = cv2.absdiff(video,videobg)
        ret,vidmask1 = cv2.threshold(vidmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,vidmask2 = cv2.threshold(vidmask2,0,255,cv2.THRESH_TRIANGLE)
        ret,vidmask3 = cv2.threshold(vidmask3,0,255,cv2.THRESH_TRIANGLE)
        vidmaskf = cv2.bitwise_and(vidmask1,vidmask2)
        vidmaskf1 = cv2.bitwise_and(vidmask1,vidmask3)

        depmask1 = cv2.absdiff(depth,depth0)
        depmask2 = cv2.absdiff(depth,depth1)
        ret,depmask1 = cv2.threshold(depmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,depmask2 = cv2.threshold(depmask2,ret,255,cv2.THRESH_BINARY)
        depmaskf = cv2.bitwise_and(depmask1,depmask2)

        fmask1 = cv2.bitwise_and(depmaskf,vidmaskf)
        fmask2 = cv2.bitwise_or(depmaskf,vidmaskf)
        fmask3 = cv2.bitwise_and(depmaskf,vidmaskf1)
        cv2.imshow('dep1',depmask1)
        cv2.imshow('vid1',vidmask1)
        cv2.imshow('vid2',vidmaskf1)
        cv2.imshow('dep',depmaskf)
        cv2.imshow('vid',vidmaskf)
        cv2.imshow('1',fmask1)
        cv2.imshow('2',fmask2)
        cv2.imshow('3',fmask3)
        cv2.waitKey(1)
        # if cv2.waitKey(1) == 27:
        #     break
    video0 = video
    depth0 = depth 



def vidframes():
    tx,td = importfiles()
    r1 = cv2.VideoCapture('myvid bs/basic.mp4')
    r2 = cv2.VideoCapture('myvid bs/dd.mp4')
    r3 = cv2.VideoCapture('myvid bs/mog.mp4')
    r4 = cv2.VideoCapture('myvid bs/ra.mp4')
    r5 = cv2.VideoCapture('myvid bs/td.mp4')
    r6 = cv2.VideoCapture('myvid bs/amf.mp4')
    i = 0
    while(r1.isOpened()):
        ret,vid = tx[0].read()
        ret,basic = r1.read()
        ret,dd = r2.read()
        ret,mog = r3.read()
        ret,ra = r4.read()
        ret,td = r5.read()
        ret,amf = r6.read()
        cv2.imwrite('myvid bs/%03dvid.png'%i,vid)
        cv2.imwrite('myvid bs/%03dbasic.png'%i,basic)
        cv2.imwrite('myvid bs/%03ddd.png'%i,dd)
        cv2.imwrite('myvid bs/%03dmog.png'%i,mog)
        cv2.imwrite('myvid bs/%03dra.png'%i,ra)
        cv2.imwrite('myvid bs/%03dtd.png'%i,td)
        cv2.imwrite('myvid bs/%03damf.png'%i,amf)
        i +=1
        
def file2():
    os.chdir('/Users/justinchan/desktop') #change to thesis directory
    i = 1915
    amf = cv2.imread('1041amf.png')
    basic = cv2.imread('1041basic.png')
    dd = cv2.imread('1041dd.png')
    gt = cv2.imread('1041gt.png')
    mog = cv2.imread('1041mog.png')
    ra = cv2.imread('1041ra.png')
    td = cv2.imread('1041td.png')
    vid = cv2.imread('1041vid.png')

    cv2.putText(vid,"RGB",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(gt,"Ground Truth",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(basic,"Basic",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(td,"TD",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(dd,"DD",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(ra,"RA",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(amf,"AMF",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    cv2.putText(mog,"MOG",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    height,width = vid.shape[:2]
    newim = np.zeros((height*4,width*2,3),dtype = "uint8")
    
    newim[0:height,0:width] = vid
    newim[0:height,width:2*width] = gt
    newim[height:2*height,0:width] = basic
    newim[height:2*height,width:2*width] = td
    newim[2*height:3*height,0:width] = dd
    newim[2*height:3*height,width:2*width] = ra
    newim[3*height:4*height,0:width] = amf
    newim[3*height:4*height,width:2*width] = mog

    newim[0:4*height,0:1] = (0,255,0)
    newim[0:4*height,2*width:2*width-1] = (0,255,0)
    newim[0:4*height,width-1:width+1] = (0,255,0)

    newim[0:1,0:2*width] = (0,255,0)
    newim[height-1:height+1,0:2*width] = (0,255,0)
    newim[2*height-1:2*height+1,0:2*width] = (0,255,0)
    newim[3*height-1:3*height+1,0:2*width] = (0,255,0)
    newim[4*height-1:4*height,0:2*width] = (0,255,0)
    cv2.imwrite('lol1.png',newim) 

