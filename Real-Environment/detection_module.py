# import classifiers
svm1 = joblib.load(<PATH-TO-SVM-MODEL>)
svm2 = ...
svm3 = ...
# put classifier into a list
svm_list = [svm1, svm2, svm3]
# flag for saving images
save = 1
# counter to save images
cont = 0
# path where to save images
PATH = ...
# size of the image
size = [<number>, <number>]

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # initialize the list of coordinates and the list of the areas
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
 
    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
		# grab the last index in the indexes list, add the index value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted) using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
 
		# delete all indexes from the index list that are in the suppression list
		idxs = np.delete(idxs, suppress)
 
	# return only the bounding boxes that were picked
    final = []
    for elem in pick:
        final.append(rects[elem])
    return final

def detection():
    
    image = getImage()
    
    # minimum_distance is calculated from the correspondent depth map
    if minimum_distance > 4000:
        return 0
       
    # calculate the size of the window in relation to the distance,the regression procedure and the total area
    window_size_b = ( <number> - minimum_distance )/ <number> ) * math.sqrt(float(len(image) * len(image[0]))/307200)
    window_size_s = window_size_b * <number>
    # compute different windows to augment the detection power
    windows_h = [[window_size_b, window_size_s], [window_size_b*1.1, window_size_s*1.1], [window_size_b*0.9, window_size_s*0.9]]
    # for every window size
    for elem in windows_h:

        # defining support vars
        a = 0
        b = a + elem[0]
        c = 0
        d = c + elem[1]

        # sliding window cycle
        while (b <= len(image)): #rows

            while (d <= len(image[0])): #columns
                # create empty matrix
                img_cropped = np.zeros((elem[0],elem[1]),dtype=np.int)

                # obtain the sub-image matrix
                img_cropped[i] = image[i+a, c:d]
                    
                # saving_img is useful for debugging
                saving_img = img_cropped.copy()
                    
                # hog needs the same image size everytime, so we need to adjust different sized images
                if elem[0] != size[0] and elem[1] != size[1]:
                    img_cropped = cv2.resize(img_cropped.astype("uint8"), (size[0],size[1]))

                # calculate sqrt of the image
                img_cropped = np.sqrt(img_cropped)
                # extract the histogram
                H = feature.hog(img_cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
                # prediction
                for i in range(len(svm_list)):
                    
                    res = svm_list[i].predict([H])
                    
                    if res == 1:
                        if i != len(svm_human)-1:
                            continue
                            
                        # if all the layers have a positive outcome, add a box in the list
                        rects.append([c,a,d,b])
                        
                        # if I want to save the sub-images, setting save to 1
                        if save == 1:
                            # if cont = -1, it will become the number after the last saved png
                            if cont == -1:
                                cont += 1
                                os.chdir(PATH)
                                for file in glob.glob("*.png"):
                                    cont += 1
                            cv2.imwrite(PATH + "my_image" + str(cont) + ".png", saving_img)
                            cont += 1
                    else:
                        break

                # updating vars
                c += <number>
                d += <number>

            a += <number>
            b += <number>
            c = 0
            d = c + elem[1]
            
    # delete redundant boxes
    rects = non_max_suppression(rects, 0.1)
    # draw boxes on the image
    for elem in rects:
        cv2.rectangle(image,(elem[0],elem[1]),(elem[2],elem[3]),(0,255,0),2)
     
    # show the final outcome
    cv2.imshow("final" + str(cont), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return the number of the identified subjects
    return len(rects)    
