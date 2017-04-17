# import necessary modules
import cv2
import freenect
import numpy as np
import time

# main function call
def depthbs():
    # read first frame for kinect sensor
    depframe = get_depth()
    vidframe = get_video()
    depthbg = depframe
    # depthbg = cv2.cvtColor(depthbg,cv2.COLOR_BGR2GRAY)
    depth0 = depthbg.copy()

    while(1):
        # read from kinect sensor
        t1 = time.time()
        depframe = get_depth()
        vidframe = get_video()
        depth = depframe
        # depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
        # compare current frame to t-1 and bg
        depmask1 = cv2.absdiff(depth,depth0)
        depmask3 = cv2.absdiff(depth,depthbg)
        # threshold difference images
        ret,depmask1 = cv2.threshold(depmask1,0,255,cv2.THRESH_TRIANGLE)
        ret,depmask3 = cv2.threshold(depmask3,0,255,cv2.THRESH_TRIANGLE)
        # OR operation
        depmaskf3 = cv2.bitwise_or(depmask1,depmask3)
        # update previous frame
        depth0 = depth 
        # show window
        cv2.imshow('foreground mask',depmaskf3)
        cv2.imshow('RGB',vidframe)
        if cv2.waitKey(1) == ord('q'):
            break
        t2 = time.time()
        tf = (t2-t1)
        print(tf)
     # clean up video
    cv2.DestroyAllWindows()
    cv2.waitKey(1)

def pretty_depth(depth):
    """Converts depth into a 'nicer' format for display
    This is abstracted to allow for experimentation with normalization
    Args:
        depth: A numpy array with 2 bytes per pixel
    Returns:
        A numpy array that has been processed with unspecified datatype
    """
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth


def pretty_depth_cv(depth):
    """Converts depth into a 'nicer' format for display
    This is abstracted to allow for experimentation with normalization
    Args:
        depth: A numpy array with 2 bytes per pixel
    Returns:
        A numpy array with unspecified datatype
    """
    return pretty_depth(depth)


def video_cv(video):
    """Converts video into a BGR format for display
    This is abstracted out to allow for experimentation
    Args:
        video: A numpy array with 1 byte per pixel, 3 channels RGB
    Returns:
        A numpy array with with 1 byte per pixel, 3 channels BGR
    """
    return video[:, :, ::-1]  # RGB -> BGR

def get_depth():
    return pretty_depth_cv(freenect.sync_get_depth()[0])

def get_video():
    return video_cv(freenect.sync_get_video()[0])

depthbs()