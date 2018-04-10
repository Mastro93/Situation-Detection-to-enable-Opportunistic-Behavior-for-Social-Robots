# initializing publisher and subscriber global vars
sub = 0
pub = 0

# initializing pca to reduce images
pca = PCA(n_components = 8)
pca10 = PCA(n_components = 10)

# importing classifiers by command joblib.load, which need absolute path of the pkl file
svm_d = joblib.load(<PATH-TO-SVM>)
svm_d1 = ...
svm_d2 = ...
svm_d3 = ...

svm_de = ...
svm_de1 = ...
svm_de2 = ...
svm_de3 = ...

svm_f = ...

svm_he = ...

# size of the sliding windows
# it is important to know the same size of the training images for the hog classifiers
# because the HOG descriptor depends strictly on the size of the image in input
# size of door handle image
size_d = [50,30]
# size of human image
size_h = [240,480]
# for handle detection
windows_d = [[30,50],[40,60]]
# support vars to speed up computation
max_row_d = 300
max_column_d = 300

# for human detection
windows_h = [[480,240]]

# method called if this node receives a "wall" message from navigate node
def callbackHandle(data):
    global sub
    global pub
    global cont

    # Transform the image to a working format
    bridge = CvBridge()
    # converting image from ROS format to a working standard 
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    # Converting to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edge detection
    # sliding rotine for every window size
    for elem in windows_d:

        # define support variables
        a, c = 0
        b = a + elem[0]
        d = c + elem[1]

        # create empty matrix
        img_cropped = np.zeros((elem[0],elem[1]),dtype=np.int)
        # Sliding window routine
        while (b <= max_row_d):

            while (d <= max_column_d):

                for i in range(elem[0]): # creating subimage
                    img_cropped[i] = gray[i+a, c:d]
                
                # reducing
                pca.fit(img_cropped)
                # svm prediction
                res = svm_d.predict([pca.singular_values_])
                if elem[0] != size_d[1]:
                    img_cropped = cv2.resize(img_cropped.astype("uint8"), (size_d[0],size_d[1]))
                # calculate sqrt of the image
                img_cropped = np.sqrt(img_cropped)
                # extract the histogram
                H = feature.hog(img_cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
                res += svm_de.predict([H])
                if res == 2: # level of the cascade

                    res1 = svm_d1.predict([pca.singular_values_])
                    res1 += svm_de1.predict([H])

                    if res1 == 2: # level of the cascade

                        res2 = svm_d2.predict([pca.singular_values_])
                        res2 += svm_de2.predict([H])

                        if res2 == 2: # level of the cascade
                        
                            # result of the classification
                            score_d += (res2/2)
                            
                # updating vars
                c += 20
                d += 20

            a += 20
            b += 20
            c = 0
            d = c + elem[1]
            
    sub.unregister()
    # create message for navigate node
    if score_d != 0:
        pub.publish("door")
        print "Door detected"
    else:
        pub.publish("nothing")
        print "No doors"
    # subscription to default waiting topic
    sub = rospy.Subscriber("/gray_request", String, callbackReq)
        
# method called if this node receives a "obstacle" message from navigate node
def callbackObstacle(data):
    global sub
    global pub

    # Transform the image to a working format
    bridge = CvBridge()
    # converting image from ROS format to a working standard 
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    # Converting to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # check for the bottom part of the image to see eventual obstacle
    img_cropped = np.zeros((150, 280),dtype=np.int)
    for i in range(len(img_cropped)):
        img_cropped[i] = gray[329+i, 100:380]
    pca.fit(img_cropped)
    res = svm_f.predict([pca.singular_values_])
    
    # if a obstacle is found
    if res == 1:
        print("Obstacle detected, analyzing...")
    
        #size of the window to detect human
        score = 0
        res = 0
        # sliding rotine for every window size
        for elem in windows_h:
            
            # initialize vars
            a = 0
            b = a + elem[0]
            c = 0
            d = c + elem[1]
            
            # initialize empty sub-image
            img_cropped = np.zeros((elem[0],elem[1]),dtype=np.int)
            while (b <= len(gray)):

                while (d <= len(gray[0])):

                    # populate sub-image
                    for i in range(elem[0]):
                        img_cropped[i] = gray[i+a, c:d]
                        
                    # svm prediction
                    if elem[0] != size_h[1]:
                        img_cropped = cv2.resize(img_cropped.astype("uint8"), (size_h[0],size_h[1]))
                    # calculate sqrt of the image
                    img_cropped = np.sqrt(img_cropped.astype("uint8"))
                    # extract the histogram
                    H = feature.hog(img_cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
                    # prediction
                    res = svm_he.predict([H])
                    # if the result is positive...
                    if res == 1:
                        # update score
                        score += 1
                    
                    # update vars
                    c += 30
                    d += 30

                a += 30
                b += 30
                c = 0
                d = c + elem[1]
	        
        sub.unregister()
        rospy.sleep(1.)
        # create message for navigate node
        if score >= 1:
            pub.publish("human")
            print "Human detected"
        else:
            pub.publish("nothing")
            print "No humans"
        sub = rospy.Subscriber("/gray_request", String, callbackReq)
    else:
        # if a malfuctioning is spotted, Turtlebot keeps going forward
        print "Malfunctioning spotted, keeping on going"
        sub.unregister()
        # message "proceed" is triggered
        pub.publish("proceed")
        # subscription to default waiting topic
        sub = rospy.Subscriber("/gray_request", String, callbackReq)
    
# method called if this node receives a "narrow" message from navigate node
def callbackNarrow(data):
    global sub
    global pub

    # no operation here yet, just publishing a message "continue" for the navigate node
    sub.unregister()
    pub.publish("continue")
    # subscription to default waiting topic
    sub = rospy.Subscriber("/gray_request", String, callbackReq)
        
# method called when a message on gray_request is published
def callbackReq(data):
    global sub
    global pub
    
    # print message received from navigate node
    print "From Navigate node: " + data.data
    # creation of a publisher node to give the results after the analysis
    pub = rospy.Publisher("gray_info", String, queue_size=10)
    sub.unregister()
    # according to the message, different methods are called
    if data.data == "wall":
        sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackHandle)
        # handle detection
        # possible outcomes: ask for help if door is 1 or turn about 90 degrees and proceed
    elif data.data == "obstacles":
        sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackObstacle)
        # human detection, malfunctioning detection
        # possible outcomes: greet human, go ahead or turn around
    elif data.data == "narrow":
        sub = rospy.Subscriber("/camera/rgb/image_raw", Image, callbackNarrow)
        # navigation with sensors
        # possible outcomes: navigate through narrow passage, then proceed forward again
    
def listener():
    global sub

    # initializing node gray-listener
    rospy.init_node('gray_listener', anonymous=True)

    # subscription to default waiting topic
    sub = rospy.Subscriber("/gray_request", String, callbackReq)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
