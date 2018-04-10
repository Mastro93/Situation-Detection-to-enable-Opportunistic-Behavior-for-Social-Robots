PATH = ...
# counter to save images
contDepth = 0
contRgb = -1
contIr = 0
# sw tool to convert images from ROS format to working formats
bridge = CvBridge()

def callbackDepth(data):

    global contDepth
    global bridge

    # Transform the image to a working format
    cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
    depth_array = np.array(cv_image, dtype=np.float32)
    
    # normalization of the distances matrix
    cv2.normalize(depth_array, depth_array, 0, 255, cv2.NORM_MINMAX)
    
    # eliminate NaN entries
    array = np.nan_to_num(depth_array)
    
    # if counter == -1, the image will have the number next to the last saved image
    if contDepth == -1:
        contDepth += 1
        os.chdir(PATH)
        for file in glob.glob("*.png"):
            contDepth += 1

    # saving image
    rospy.loginfo(str(cv2.imwrite(PATH + "my_image" + str(contDepth) + ".png", array)) + " depth")
    
    contDepth += 1
    
def callbackDepthRaw(data):

    global contDepth
    global bridge

    # Transform the image to a working format
    cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
    depth_array = np.array(cv_image, dtype=np.float32)
    
    # if counter == -1, the image will have the number next to the last saved image
    if contDepth == -1:
        contDepth += 1
        os.chdir(PATH)
        for file in glob.glob("*.txt"):
            contDepth += 1
    
    # write the matrix of distances on a file
    fil = open(PATH + "my_image" + str(contDepth) + ".txt", "w")    
    for i in range(len(depth_array)):
        for j in range(len(depth_array[0])):
            fil.write(str(depth_array[i][j]) + "\n")
    fil.close()
    
    contDepth += 1

def callbackRgb(data):
    
    global contRgb
    global bridge
    
    # convert to a working bgr8 format
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    # convert in grayscale (if needed)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # if counter == -1, the image will have the number next to the last saved image
    if contRgb == -1:
        contRgb += 1
        os.chdir(PATH)
        for file in glob.glob("*.png"):
            contRgb += 1
            
    # saving the image
    rospy.loginfo(str(cv2.imwrite(PATH + "my_image" + str(contRgb) + "-rgb.png", gray)) + " rgb")
    contRgb += 1
    
def callbackIr(data):
    global contIr
    global bridge

    # convert to a working format
    img = bridge.imgmsg_to_cv2(data, "8UC1")
    ir_array = np.array(img, dtype=np.float32)
    # normalization
    cv2.normalize(ir_array, ir_array, 0, 255, cv2.NORM_MINMAX)
    
    # if counter == -1, the image will have the number next to the last saved image
    if contIr == -1:
        contIr += 1
        os.chdir(PATH)
        for file in glob.glob("*.png"):
            contIr += 1
    
    # save the image
    rospy.loginfo(str(cv2.imwrite(PATH + "my_image" + str(contIr) + "-ir.png", ir_array)) + " ir")
    contIr += 1

def saver():

    rospy.init_node('saver', anonymous=True)

    # uncomment the image topic you want to subscribe
    #rospy.Subscriber("/camera/depth/image_raw", Image, callbackDepth)
    #rospy.Subscriber("/camera/depth/image_raw", Image, callbackDepthRaw)
    #rospy.Subscriber("/camera/rgb/image_raw", Image, callbackRgb)
    #rospy.Subscriber("/camera/ir/image", Image, callbackIr)    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    saver()
