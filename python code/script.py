# uncomment function and python script.py to run
from test import *
[tx,td] = importfiles()
f = open('bsspeed.txt','a')

# vidframes()
# file2()
# approximatemedian(tx[2])
# temporaldifferencing(tx[2])
# runningaverage(tx[2])
# mog(tx[2])
# depth(td[4]

# depth = cv2.VideoCapture('depth results/depthresults.mp4')
# tx1 = tx[0]
# td1 = td[0]
# i = 0
# while tx1.isOpened():
# 	ret,tx11 = tx1.read()
# 	ret,td11 = td1.read()
# 	ret,depthh = depth.read()
# 	cv2.imwrite('bs/%03dvid.png' % i,tx11)
# 	cv2.imwrite('bs/%03ddep.png' % i,td11)
# 	cv2.imwrite('bs/%03dres.png' % i,depthh)
# 	i+=1

os.chdir('/Users/justinchan/desktop') #change to thesis directory
dep = cv2.imread('773dep.png')
res = cv2.imread('773res.png')
vid = cv2.imread('773vid.png')
gt =cv2.imread('773gt.png')

height,width = vid.shape[:2]
cv2.putText(vid,"RGB",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
cv2.putText(gt,"Ground Truth",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
cv2.putText(dep,"Depth",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
cv2.putText(res,"Foreground Mask",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

newim = np.zeros((height*2,width*2,3),dtype = "uint8")

newim[0:height,0:width] = vid
newim[0:height,width:2*width] = dep
newim[height:2*height,0:width] = res
newim[height:2*height,width:2*width] = gt

newim[0:1,0:2*width] = (0,255,0)
newim[height-1:height+1,0:2*width] = (0,255,0)
newim[2*height-1:2*height,0:2*width] = (0,255,0)
newim[0:2*height,0:1] = (0,255,0)
newim[0:2*height,width-1:width+1] = (0,255,0)
newim[0:2*height,2*width-1:2*width] = (0,255,0)

cv2.imwrite('773b.png',newim)
    