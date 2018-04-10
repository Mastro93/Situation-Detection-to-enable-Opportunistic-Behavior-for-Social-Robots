# folder for saving images
PATH = ...
# camera publishes a lot of images, use the counter to select how many images you want to see
cont_image = -1
# initialize vars
pub = 0
# distance threshold
threshold_w = 1.5

def callback(data):
    global cont_image
    global pub
    cont_image += 1
    # one image every 75 is analyzed
    if cont_image % 75 == 0:
        global cont

        # Transform the image to a working format
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
        depth_array = np.array(cv_image, dtype=np.float32)
        array = np.nan_to_num(depth_array)
        # I want to identify passages and obstacles in front of the agent defining a list of
        # clear zones based on the indexes in the depth map
        passage = []
        start_index = 0
        length = 0
        # kind of obstacles
        wall = 1
        narrow = 0
        obstacles = 0
        # selection of the middle raw of the depth map
        for i in range(len(array[0])):
            entry = array[len(array)/2][i]
            # if something has distance 0.0 (out of maximum range) or bigger than a threshold
            # then there is not a wall
            if entry == 0.0 or entry > threshold_w:
                wall = 0
                if length == 0:
                    start_index = i
                length += 1
            elif length != 0:
                passage.append([start_index, length])
                length = 0
                start_index = 0
            elif obstacles == 1:
                pass
            # I consider obstacle only an entity that would hinder the agent
            elif i > 100 and i < 380:
                obstacles = 1
                
        if length != 0:
            passage.append([start_index, length])
                
        # detection of narrow passages
        if len(passage) != 0:
            for i in range(len(passage)):
                start_index = passage[i][0]
                length = passage[i][1]
                if length > 150 and length < 300 and start_index != 0 and start_index+length != len(array[0]):
                    narrow = 1
            
        # making up false obstacles to see if floor svm works
        rand = np.random.rand()
        if obstacles == 0 and rand > 0.92:
            obstacles = 1
           
        # based on the vars values, publishing different messages
        if wall == 1:
            rospy.loginfo("Wall detected")
            pub.publish("wall")
        elif narrow == 1:
            rospy.loginfo("Narrow passage detected")
            pub.publish("narrow")
        elif obstacles == 1:
            rospy.loginfo("Obstacle detected")
            pub.publish("obstacles")
        else:
            rospy.loginfo("Nothing detected")
            pub.publish("")
        
def listener():
    global pub
    
    # initializing depth-listener
    rospy.init_node('depth-listener', anonymous=True)

    # subscribe to depth image topic
    rospy.Subscriber("/camera/depth/image_raw", Image, callback)

    # initializing depth_info topic to publish information about obstacles
    pub = rospy.Publisher('depth_info', String, queue_size=10)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
