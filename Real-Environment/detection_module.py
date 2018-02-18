#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
import sklearn
from sklearn.externals import joblib
import scipy.ndimage
import scipy.misc
from sklearn.decomposition import PCA
import imutils
from skimage import exposure
from skimage import feature
import glob, os
import math
from scipy.ndimage import morphology

# importing svm models
svm_dsqrt1 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/real-diag-handle-hog-sqrt-pyr.pkl")
svm_dsqrt2 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/cascade/real-diag-handle-hog-sqrt-pyr-cascade.pkl")
svm_dsqrt3 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/cascade/cascade1/real-diag-handle-hog-sqrt-pyr-cascade1.pkl")
svm_dsqrt4 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/cascade/cascade1/cascade2/real-diag-handle-hog-sqrt-pyr-cascade2.pkl")

svm_d1 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/real-diag-handle-pyr.pkl")
svm_d2 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/cascade/real-diag-handle-pyr-cascade.pkl")
svm_d2_2 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/cascade/real-diag-handle-cascade.pkl")
svm_d3 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-diag-handle-hog/cascade/cascade1/real-diag-handle-pyr-cascade1.pkl")

#svm_d1 = joblib.load("/home/daniele/image_transport_ws/src/image_transport_tutorial/include/image_transport_tutorial/real-diag-handle.pkl")
#svm_d2 = joblib.load("/home/daniele/image_transport_ws/src/image_transport_tutorial/include/image_transport_tutorial/real-diag-handle-cascade.pkl")

#svm_de1 = joblib.load("/home/daniele/image_transport_ws/src/image_transport_tutorial/include/image_transport_tutorial/real-diag-handle-pyr-edge450.pkl")
#svm_de2 = joblib.load("/home/daniele/image_transport_ws/src/image_transport_tutorial/include/image_transport_tutorial/real-diag-handle-pyr-edge450-cascade.pkl")

svm_dh = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-human-legs-side-pyr/real-human-legs-side-hog-sqrt-pyr.pkl")
svm_dh1 = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-human-legs-side-pyr/cascade/real-human-legs-side-hog-sqrt-pyr-cascade.pkl")

svm_fl = joblib.load("/home/daniele/generatore_eventi/real-env-footage/work-floor/sun-floor-depth.pkl")

# subscribers
sub = 0
# save flag
save = 1
# saving path
PATH = "/home/daniele/generatore_eventi/real-env-footage/test_xtion/"
# pca
pca = PCA(n_components = 10)
# initiliazing distance
minimum_distance = 10000.0
# size classifier image for human legs
size_h = [75,101]
# size classifier image for handle
size_d = [85,75]
# bridge to convert images
bridge = CvBridge()
# initialize list of boxes
rects = []
# counters
cont = -1
# images acquired
GRAY = np.zeros((640,480),dtype=np.int)
depthMap = np.zeros((640,480),dtype=np.int)
# flag 
images_ok = 0


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([ ((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    x1 = []
    y1 = []
    x2 = []
    y2 = []    
    area = []
	# grab the coordinates of the bounding boxes
    for i in range(len(boxes)):
        x1.append(boxes[i][0])
        y1.append(boxes[i][1])
        x2.append(boxes[i][2])
        y2.append(boxes[i][3])
        area.append((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1))
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
 
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
 
	# return only the bounding boxes that were picked
    final = []
    for elem in pick:
        final.append(rects[elem])
    return final
    
   
def callbackDistance(): # calculate the distance from the closest object in front of the camera
    global minimum_distance
    
    depth_array = depthMap.copy()
    
    # Select the central row
    row = depth_array[len(depth_array)/2]
    # Save row length
    row_length = len(depth_array[0])
    # Initialize minimum distance with a high number
    minimum_distance = 10000.0
    
    # select the closest point
    for i in range(row_length):
        if i != 0 and float(i)/float(row_length) >= 0.25 and float(i)/float(row_length) <= 0.75:
            if row[i] < minimum_distance and row[i] != 0.0:
                minimum_distance = row[i]
                
    # inform about what is the actual minimum distance
    print "The closer object is at " + str(minimum_distance) + "mm from the agent"

def callbackFace():
    
    gray = GRAY.copy()
    
    # retrieve the cascade classifier for faces
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    # detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # draw boxes
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)

    # show the image
    cv2.imshow('final', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # return the number of faces
    return len(faces)

def callbackHandle():
    global rects
    global cont
    
    gray = GRAY.copy()

    # exploit the relation to set the optimale size of the sliding window
    window_min_size = float( (1340.0 - float(minimum_distance/10.0) * float(3)) /float(11) ) * math.sqrt(float(len(gray)*len(gray[0]))/float(307200))
    window_big_size = window_min_size * float(float(size_d[0])/float(size_d[1]))
    windows_d = [[int(window_min_size), int(window_big_size)]]
    print windows_d
    for elem in windows_d:

        # defining support vars
        a = len(gray)/4 # only the central part of the image is scanned for computation purposes
        b = a + elem[0]
        c = len(gray[0])/4
        d = c + elem[1]

        # sliding window cycle
        while (b <= 3*len(gray)/4): # rows

            while (d <= 3*len(gray[0])/4): # columns
                # create empty matrix
                img_cropped = np.zeros((elem[0],elem[1]),dtype=np.int)

                # extract the sub-image
                img_cropped = gray[a:b, c:d]                    
                
                # keep the original sub-image
                saving_img = img_cropped.copy()
                
                # hog needs the same image size everytime, so we need to adjust different sized images
                if elem != [size_d[1],size_d[0]]:
                    img_cropped = cv2.resize(img_cropped.astype("uint8"), (size_d[0],size_d[1]))
                
                # calculate sqrt of the image
                img_cropped = np.sqrt(img_cropped.astype("uint8"))
                # extract the histogram
                H = feature.hog(img_cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
                # reduce the original sub-image
                pca.fit(saving_img)
                # predict with the couple of svms 
                res = svm_d1.predict([pca.singular_values_])
                res += svm_dsqrt1.predict([H])

                # if both the predicted labels are 1
                if res == 2:

                    res1 = svm_dsqrt2.predict([H])

                    if res1 == 1:

                        res2 = svm_dsqrt3.predict([H])
                        
                        if res2 == 1:
                        
                            res3 = svm_dsqrt4.predict([H])
                            
                            if res3 == 1:
                            
                                # if all the layers have a positive outcome, add a box in the list
                                rects.append([c,a,d,b])
                                # if I want to save the sub-images, setting save to 1
                                if save == 1:
                                    if cont == -1:
                                        cont += 1
                                        os.chdir(PATH)
                                        for file in glob.glob("*.png"):
                                            cont += 1
                                    cv2.imwrite(PATH + "my_image" + str(cont) + ".png", saving_img)
                                    cont += 1

                # updating vars
                c += 15
                d += 15

            a += 15
            b += 15
            c = 0
            d = c + elem[1]

    # delete redundant boxes                 
    rects = non_max_suppression_slow(rects, 0.1)
    # draw boxes on the image
    for elem in rects:
        cv2.rectangle(gray,(elem[0],elem[1]),(elem[2],elem[3]),(0,255,0),2)
        
    # show the whole image with the boxes
    cv2.imshow("final", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # wait for 5 seconds
    rospy.sleep(5.)
    # return the number of handles
    return len(rects)    
    
def callbackHuman():
    global cont
    global rects
    
    gray = GRAY.copy()
       
    # exploit the relation to set the optimale size of the sliding window
    window_big_size = float( float( 970.0 - float(minimum_distance/10.0) )/2 ) * math.sqrt(float(len(gray)*len(gray[0]))/307200)
    window_min_size = window_big_size * 0.74
    windows_h = [[int(window_big_size), int(window_min_size)]]
    print windows_h
    # for every window size
    for elem in windows_h:

        # defining support vars
        a = 0
        b = a + elem[0]
        c = 0
        d = c + elem[1]

        # sliding window cycle
        while (b <= len(gray)): #rows

            while (d <= len(gray[0])): #columns
                # create empty matrix
                img_cropped = np.zeros((elem[0],elem[1]),dtype=np.int)

                # extract the sub-image
                img_cropped = gray[a:b, c:d]
                    
                # keep the original sub-image
                saving_img = img_cropped
                    
                # hog needs the same image size everytime, so we need to adjust different sized images
                if elem[0] != size_h[1]:
                    img_cropped = cv2.resize(img_cropped.astype("uint8"), (size_h[0],size_h[1]))

                # calculate sqrt of the image
                img_cropped = np.sqrt(img_cropped)
                # extract the histogram
                H = feature.hog(img_cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
                # prediction
                res = svm_dh.predict([H])
                
                if res == 1:

                    res1 = svm_dh1.predict([H])

                    if res1 == 1:
                        # if I want to save the sub-images, setting save to 1
                        if save == 1:
                            if cont == -1:
                                cont += 1
                                os.chdir(PATH)
                                for file in glob.glob("*.png"):
                                    cont += 1
                            cv2.imwrite(PATH + "my_image" + str(cont) + ".png", saving_img)
                            cont += 1
                        # if all the layers have a positive outcome, add a box in the list
                        rects.append([c,a,d,b])

                # updating vars
                c += 20
                d += 20

            a += 20
            b += 20
            c = 0
            d = c + elem[1]
            
    # delete redundant boxes
    rects = non_max_suppression_slow(rects, 0.1)
    # draw boxes on the image
    for elem in rects:
        cv2.rectangle(gray,(elem[0],elem[1]),(elem[2],elem[3]),(0,255,0),2)
     
    # show the final outcome
    cv2.imshow("final" + str(cont), gray)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    # sleep for 2 seconds
    rospy.sleep(2.)

    # return the number of human legs
    return len(rects)
    
def callbackSun():
    
    depth = depthMap.copy()

    # normalizing depth map
    cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX)
    # nan_to_num to substitute NaN entries
    depth = np.nan_to_num(depth)
    # changing format
    depth = depth.astype("uint8")
    # reduction
    pca.fit(depth)
    # return the prediction
    return svm_fl.predict([pca.singular_values_])

def callbackDepth(data): # if you have ROS, keep this method to retrieve the image from the camera
    global depthMap
    global images_ok
    global sub

    # Transform the image to a working format
    cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
    depthMap = np.array(cv_image, dtype=np.float32)
    
    print "Depth map acquired"
    
    sub.unregister()
    # flag needed to confirm that all the images have been retrieved
    images_ok = 1
    
def callbackRGB(data): # if you have ROS, keep this method to retrieve the image from the camera
    global GRAY
    global sub

    # Transform the image to a working format
    bridge = CvBridge()
    # converting image from ROS format to a working standard 
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    # Converting to gray-scale
    GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print "RGB image acquired"
    
    sub.unregister()
    sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, callbackDepth)
    
def listener():
    global sub
    global rects
    global images_ok
    
    
    # initializing listener
    rospy.init_node('stillListenerMoveV2', anonymous=True)

    # subscribe to image topics
    sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackRGB)
    
    # Loop until all the images have been retrieved
    while images_ok == 0:
        pass
        
    # re-initialize the flag 
    images_ok = 0
    # the procedure is repeated in order to take the current shots and not the ones in the buffer
    sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackRGB)
    
    while images_ok == 0:
        pass

    # call the methods you want to use
    callbackDistance() # set the minimum distance
    #legs = callbackHuman() # search for human legs
    rects = [] # re-initialize the boxes list
    #face = callbackFace() # search for human faces
    rects = [] # re-initialize the boxes list
    handle = callbackHandle() # search for door handles
    rects = [] # re-initialize the boxes list
    #sun = callbackSun() # search for projected sunlight

    # in according of the methods called, uncomment these prints to see the results
    #print "legs: " + str(legs)
    #print "face: " + str(face)
    #print "handle: " + str(handle)
    #print "sun: " + str(sun)
    
    # re-initialize the flag
    images_ok = 0
    
if __name__ == '__main__':
    # the method will terminate when Ctrl + C is pressed
    while not rospy.is_shutdown():
        listener()
