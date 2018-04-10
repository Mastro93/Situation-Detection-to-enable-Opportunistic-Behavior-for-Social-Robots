# number of total images
NUM_IMG = ...
# support variables
N_END_A = ...
N_END_B = ...
N_END_C = ...

# open a txt file
labels_file = open("labels.txt", "w")

# write in every row the label of the correspondent image
# help yourself with the support variables
for i in range(NUM_IMG):
    if i <= N_END_A:
        labels_file.write("1\n")
    else:
        labels_file.write("0\n")

labels_file.close()
