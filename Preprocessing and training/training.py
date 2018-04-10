# number of total images
NUM_IMGS = ...
# number of components of the image vector
N_COMPS = ...
# shuffle the samples during training
SHUFFLED = ...
# kernel of the SVM
KERNEL = "linear"
# HOG or not
HOG = ...
# name of the trained model
NOMEFILE = "real-human-legs-side-hog-pca-sqrt-pyr.pkl"

def dim_red_shuffled():
    print("Preprocessing...")

    # create an empty input array
    array = np.zeros(shape=(NUM_IMGS,N_COMPS))
    # create a list
    indexes = [i for i in range(NUM_IMGS)]
    # shuffle such list
    shuffle(indexes)
    # initialize PCA
    pca = PCA(n_components = N_COMPS)

    for i in range(NUM_IMGS):
        img = cv2.imread("my_image" + str(indexes[i]) + ".png", 0)
        if HOG != 0:
            # the length of the HOG descriptor depends on the size of the image
            # therefore, it has to be always the same
            img = cv2.resize(img, (75,101))
            # apply the square root normalization
            img = np.sqrt(img)
            # calculate HOG descriptor
            H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
            # put it in the input matrix
            array[i] = H
        else:
            # calculate the vector of principal components
            pca.fit(img)
            # put such vector in the input matrix
            array[i] = pca.singular_values__

    print("Done.")
    return array, indexes

def dim_red():
    print("Preprocessing...")

    # create an empty input matrix
    array = np.zeros(shape=(NUM_IMGS,N_COMPS))
    # initialize PCA
    pca = PCA(n_components = N_COMPS)
    
    for i in range(NUM_IMGS):
        img = cv2.imread("my_image" + str(i) + ".png", 0)
        if HOG != 0:
            # the length of the HOG descriptor depends on the size of the image
            # therefore, it has to be always the same
            img = cv2.resize(img, (75,101))
            # apply the square root normalization
            img = np.sqrt(img)
            # calculate HOG descriptor
            H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
            # put it in the input matrix
            array[i] = H
        else:
            # calculate the vector of principal components
            pca.fit(img)
            # put such vector in the input matrix
            array[i] = pca.singular_values_
            
    print("Done.")
    return array
    
def supervised():
    
    # import labels, creating two identical list for shuffling
    print("\n########\nEXECUTING\nImporting labels...")
    label_file = open("labels.txt", "r")
    labels = []
    labels2 = []
    var = 0
    for i in range(NUM_IMGS*4):
        var = int(label_file.readline().replace("\n", ""))
        labels.append(var)
        labels2.append(var)
    label_file.close()
    print("Done.")
    
    # initialize SVM
    svc = svm.SVC(kernel=KERNEL)
    array = []
    shuffled = []
    # call to functions that create the input matrix
    if SHUFFLED:
        array, shuffled = dim_red_shuffled()
        for i in range(NUM_IMGS):
            labels[i] = labels2[shuffled[i]]
    else:
        array = dim_red()

    # training phase, this may take a bunch of minutes
    print("Model creation...")
    svc.fit(array, labels)
        
    # saving trained model
    joblib.dump(svc, NOMEFILE)
    
if __name__ == "__main__":
    print("Number of images = " + str(NUM_IMGS) + "\nNumber of vector components = " + str(N_COMPS)+ "\nShuffled = " + str(SHUFFLED) + "\nKernel = " + KERNEL + "\nHOG = " + str(HOG))
    supervised()
