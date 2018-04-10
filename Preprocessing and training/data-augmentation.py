# open an image
img = cv2.imread(<PATH-TO-IMAGE>, 0)

# flip the image vertically
img_flipped1 = img[:,::-1]
# flip the iimage orizontally
img_flipped2 = img[::-1,:]

# save the flipped images
cv2.imwrite("flipped1.png", img_flipped1)
cv2.imwrite("flipped2.png", img_flipped2)

# rotate the image of 5°
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols*3/4,rows*3/4),5,1)
dst = cv2.warpAffine(img,M,(cols,rows))

# save it
cv2.imwrite("rotated5.png", dst)

# rotate the image of -5°
M = cv2.getRotationMatrix2D((cols*3/4,rows*3/4),-5,1)
dst = cv2.warpAffine(img,M,(cols,rows))

# save it
cv2.imwrite("rotate-5.png", dst)

# calculate the dilated and the reduces version of the original sample
imgUp = cv2.pyrUp(img)
imgDown = cv2.pyrDown(img)

# save them
cv2.imwrite("up.png", imgUp)
cv2.imwrite("down.png", imgDown)
