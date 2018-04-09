# save images in
PATH = ...
# flag for saving images
save = 0
# count to save images
cont = -1

depth_image = getDepthImage()
ir_image = getInfraredImage()
# compute edged images
depth_edge = imutils.auto_canny(depth_image)
ir_edge = imutils.auto_canny(ir_image)
# dilation of lines
depth_edge = morphology.binary_dilation(depth_edge, iterations = 5)
ir_edge = morphology.binary_dilation(ir_edge, iterations = 5)
# erosion of lines
depth_edge = morphology.binary_erosion(depth_edge)
ir_edge = morphology.binary_erosion(ir_edge)
# transform into the right format
depth_edge = edge.astype("uint8")*255
ir_edge = ir_edge.astype("uint8")*255

final = cv2.bitwise_and(depth_edge, ir_edge)

windows = [[<number>,<number>], ...]

for elem in windows:

    # defining support vars
    a = 0
    b = a + elem[0]
    c = 0
    d = c + elem[1]
    # initialize list for coordinates and number of white pixels
    list_coord = []
    list_num = []

    # sliding window cycle
    while (b <= len(final)): # rows

        while (d <= len(final[0])): # columns
            # create empty matrix
            img_cropped = np.zeros((elem[0],elem[1]),
                          dtype=np.int)

            # populate the matrix
            img_cropped = final[a:b,c:d]
            # calculate the number of non-zero elements
            # in the sub-image
            ones = np.count_nonzero(img_cropped)
                    
            # if flag save is set to 1, sub-images are saved
            if save == 1:
                if cont == -1:
                    cont += 1
                    os.chdir(PATH)
                    for file in glob.glob("*.png"):
                        cont += 1
                cv2.imwrite(PATH + "my_image" + str(cont)
                            + ".png", img_cropped)
                cont += 1
                    
            # update list of coords and number
            # of non-zero elems
            lista_coord.append([c,a,d,b])
            lista_num.append(ones)
            
            # updating vars to scan the image
            c += 20
            d += 20

        a += 20
        b += 20
        c = 0
        d = c + elem[1]

# create a dictionary with number of
# non-zero elems as keys and coords as values
d = dict(zip(lista_num,lista_coord))
# initialize list of boxes
rects = []

# take only the 5 sub-images with the
# highest numbers of non-zero elements
for i in range(5):
    maximum = np.amax(lista_num)
    lista_num.remove(maximum)
    rects.append(d[maximum])

# delete redundant boxes                 
rects = non_max_suppression_slow(rects, 0.5)
# initialize final coords
coords = [10000,10000,0,0]
# calculate a bounding box that contain
# all the boxes found
for elem in rects:
    if elem[0] < coords[0]:
        coords[0] = elem[0]
    if elem[1] < coords[1]:
        coords[1] = elem[1]
    if elem[2] > coords[2]:
        coords[2] = elem[2]
    if elem[3] > coords[3]:
        coords[3] = elem[3]
        
# draw boxes on the image
cv2.rectangle(depth_image,(coords[0],coords[1]),(coords[2],coords[3]),(0,255,0),2)
