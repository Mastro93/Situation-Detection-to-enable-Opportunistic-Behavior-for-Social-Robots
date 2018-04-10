# if True, the robot will not move
stop = False
# support variable for the subscriber
sub = 0

class Navigate():
    def __init__(self):
        # initiliaze
        rospy.init_node('Navigate', anonymous=False)

	    # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)
        
        # Call to navigation method
        self.goForward()
        
        # spin (which means no op), otherwise node will terminate
        rospy.spin()
        
        
    def goForward(self):
        global stop
        global sub
        
	    # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
     
	    # TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10)

        # Twist is a datatype for velocity
        move_cmd = Twist()
	    # let's go forward at 0.2 m/s
        move_cmd.linear.x = 0.2
	    # let's turn at 0 radians/s
        move_cmd.angular.z = 0

        # Aggiungo la sottoscrizione al topic depth_info per bloccare eventualmente la navigazione
        sub = rospy.Subscriber("/depth_info", String, self.callbackDepth)

	    # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown() and not stop:
	        # publish the velocity
            self.cmd_vel.publish(move_cmd)
	        # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()
            
    def spin(self):
        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
     
	    # TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10)

        # Twist is a datatype for velocity
        move_cmd = Twist()
	    # let's go forward at 0.2 m/s
        move_cmd.linear.x = 0
	    # let's turn at 0 radians/s
        move_cmd.angular.z = 0.2

	    # as long as you haven't ctrl + c keeping doing...
        for _ in range(80):
	        # publish the velocity
            self.cmd_vel.publish(move_cmd)
	        # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()
            
    def avoidObstacle(self):
        #tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("wait for the action server to come up")
        #allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(10))
        print "after action server"
	    
        #we'll send a goal to the robot to move 3 meters forward
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'base_link'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = 3.0 #3 meters
        goal.target_pose.pose.orientation.w = 1.0 #go forward

        #start moving
        self.move_base.send_goal(goal)
        
        #allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(60)) 


        if not success:
            self.move_base.cancel_goal()
            rospy.loginfo("The base failed to move forward 3 meters for some reason")
        else:
            # We made it!
            state = self.move_base.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Hooray, the base moved 3 meters forward")
        
        
    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
	    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
	    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)
        
        
    def callbackDepth(self, data):
        global stop
        global sub
        # if there is something in front of the turtlebot, then I write to the SVM node in order to recognize a situation
        # and perform a specific action
        if data.data != "":
            print "Depth-Listener says: " + data.data
            stop = True
            # publishing request, it will be read by SVM node
            pub = rospy.Publisher('gray_request', String, queue_size=10)
            rospy.sleep(1.)
            pub.publish(data.data)
            # depth_info is useless now, subscription to SVM node
            sub.unregister()
            sub = rospy.Subscriber("/gray_info", String, self.callbackRGB)
                
                
    def callbackRGB(self, data):
        global stop
        global sub
        # print of information received, now I have to choose what action to perform
        print "SVM says: " + data.data
        stop = False
        # I don't need rgb info anymore
        sub.unregister()
        if data.data == "human":
            rospy.loginfo("Hi! Please, go ahead!")
            self.spin()
        elif data.data == "door":
            rospy.loginfo("May someone open this door, please?")
            self.spin()
        elif data.data == "continue":
            self.avoidObstacle()
        elif data.data == "proceed":
            pass
        else:
            self.spin()
        self.goForward()
            

if __name__ == '__main__':

    Navigate()
