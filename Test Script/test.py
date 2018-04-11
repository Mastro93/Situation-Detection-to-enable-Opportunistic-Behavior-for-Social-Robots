# number of the first image to test
FIRST_IMG = ...
# number of the last image
LAST_IMG = FIRST_IMG+1
# initialize pca
pca = PCA(n_components = 10)
# what image to analyze
SRC = "/home/daniele/generatore_eventi/real-env-footage/human/leg2/my_image"
# importing svms model
svm_dh = joblib.load("<PATH-TO-SVM>")
svm_dh1 = ...

svm_p = ...
svm_p1 = ...

# support vars
save = 0 # save sub-images if 1
cont = -1 # counter of the sub-images
# size of the sliding window
windows = [[xxx,yyy], ...]
# size of the image according to training phase
size = [kkk, zzz]
# initialize list of boxes
rects = []
# saving path
PATH = ...

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
	
if __name__ == "__main__":

    for i in range(FIRST_IMG, LAST_IMG):
    
        # select an image and open it in gray-scale
        # if there is no image, pass to the next number
        try:
            Image.open(SRC + str(i) + "-rgb.png")
        except:
            continue
        gray = cv2.imread(SRC + str(i) + "-rgb.png", 0)
        # initialize score
        score_d = 0
        # for every window size
        for elem in windows:

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
                    # select the sub-image
                    img_cropped = gray[a:b, c:d]
                    # copy the original sub-image
                    saving_img = img_cropped.copy()
                        
                    # hog needs the same image size everytime, so we need to adjust different sized images
                    if elem != [size[1],size[0]]:
                        img_cropped = cv2.resize(img_cropped.astype("uint8"), size)
                        
                    # calculate sqrt of the image
                    img_cropped = np.sqrt(img_cropped)
                    # extract the histogram
                    H = feature.hog(img_cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
                    # prediction
                    res = svm_dh.predict([H])
                    if res == 1:

                        res1 = svm_dh1.predict([H])

                        if res1 == 1:

                            # if all the layers have a positive outcome, add a box in the list
                            rects.append([c,a,d,b])
                            
                    # updating vars
                    c += 50
                    d += 50

                a += 50
                b += 50
                c = 0
                d = c + elem[1]
                     
                     
        # delete redundant boxes
        rects = non_max_suppression_slow(rects, 0.1)
        # draw boxes on the image
        for elem in rects:
            cv2.rectangle(gray,(elem[0],elem[1]),(elem[2],elem[3]),(0,255,0),2)
            
        # if I want to save the sub-images, setting save to 1
        if save == 1:
            if cont == -1:
                cont += 1
                os.chdir(PATH)
                for file in glob.glob("*.png"):
                    cont += 1
            cv2.imwrite(PATH + "test_image" + str(cont) + ".png", gray)
            cont += 1
        
    # show the final outcome
    cv2.imshow("final", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows
