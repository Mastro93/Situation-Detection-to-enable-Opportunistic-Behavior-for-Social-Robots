def callbackDistance():
    global minimum_distance
    
    depth_array = depthMap
    
    # area of the region of interest
    rows_interval = 10
    columns_interval = 10
    
    # Select the portion of image with the desired target
    region = depth_array[ (len(depth_array)-10)/2 : (len(depth_array)+10)/2, (len(depth_array[0])-10)/2 : (len(depth_array[0])+10)/2 ]
    # I want to extract the minimum distance of the object in front of the agent
    minimum_distance = 10000.0
    
    for i in range(len(region)):
        for j in range(len(region[0])):
            if region[i,j] < minimum_distance and region[i,j] != 0.0:
                minimum_distance = region[i,j]
                
    print "The closer object is at " + str(minimum_distance) + "mm from the agent"
