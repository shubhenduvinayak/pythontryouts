import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test_images/test4.jpg')

cv2.imshow("original",img)

img_h,img_w,img_c = img.shape

w_min = img_w * 0.05
w_max = img_w * 0.95

h_min = img_h * 0.05
h_max = img_h * 0.95

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (int(w_min),int(h_min),int(w_max),int(h_max))

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.colorbar()
plt.show()