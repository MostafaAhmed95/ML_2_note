import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2 as cv
import sklearn as sklearn
from sklearn import preprocessing
from PIL import Image
import random
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels
images,labels=readTrafficSigns(r'GTSRB\Final_Training\Images')


#print(len(images[0]))
#print(len(images[0][0]))

for i in range(len(images)):
    images[i] = np.asarray(images[i])
    if(images[i].shape[0]>images[i].shape[1]):
        d=images[i].shape[0]-images[i].shape[1]
        if(d % 2 == 0):
            images[i] = cv.copyMakeBorder(images[i], 0, 0, int(d/2), int(d/2), cv.BORDER_REPLICATE)
        else:
            images[i] = cv.copyMakeBorder(images[i], 0, 0, int((d-1)/2), int(((d-1)/2)+1), cv.BORDER_REPLICATE)
    elif (images[i].shape[0] < images[i].shape[1]):
        d = images[i].shape[1] - images[i].shape[0]
        if (d % 2 == 0):
            images[i] = cv.copyMakeBorder(images[i], int(d/2), int(d/2), 0, 0, cv.BORDER_REPLICATE)
        else:
            images[i] = cv.copyMakeBorder(images[i], int((d - 1)/2), int(((d - 1)/2)+1), 0, 0, cv.BORDER_REPLICATE)
    #images[i]=np.resize(images[i],(30,30,3))

#plt.imshow(images[18])
#plt.show()

#Shuffle our data
random.shuffle(images)
#Spliting our data
train_set=images[ :int(len(images)*0.8)]
test_set=images[int(len(images)*0.8): ]
train_set_np=np.asarray(train_set)
#start plotting y data
unique_elements, counts_elements = np.unique(labels, return_counts = True)
#print(np.asarray((unique_elements, counts_elements)))

#plot my data after spliting
plt.bar( np.arange( 43 ), counts_elements, align='center',color='blue' )
plt.xlabel('Class')
plt.ylabel('No of Training data')
plt.xlim([-1, 43])
#plt.show()

#data_augmentaion
def data_augment(image):
    rows = image.shape[0]
    cols = image.shape[1]

    # rotation
    M_rot = cv.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)

    # Translation
    M_trans = np.float32([[1, 0, 3], [0, 1, 6]])

    img = cv.warpAffine(image, M_rot, (cols, rows))
    img = cv.warpAffine(img, M_trans, (cols, rows))
    # img = cv2.warpAffine(img,M_aff,(cols,rows))

    # Bilateral filtering
    img = cv.bilateralFilter(img, 9, 75, 75)
    return img

classes = 43
labels=np.asarray(labels, dtype=np.int)
images=np.asarray(images)
X_train_final = train_set
y_train_final = labels
X_aug_1 = []
Y_aug_1 = []
#print(labels)
for i in range(0, classes):

    class_records = np.asarray(np.where(labels == i)).size
    max_records = 3000
    if class_records != max_records:
        ovr_sample = max_records - class_records
        samples = images[np.where(labels == i)[0]]
        X_aug = []
        Y_aug = [i] * ovr_sample

        for x in range(ovr_sample):
            img = samples[x % class_records]
            trans_img = data_augment(img)
            X_aug.append(trans_img)

        X_train_final = np.concatenate((X_train_final, X_aug), axis=0)
        y_train_final = np.concatenate((y_train_final, Y_aug))

        Y_aug_1 = Y_aug_1 + Y_aug
        X_aug_1 = X_aug_1 + X_aug

#plotting the data after agumentaion
unique_elements, counts_elements = np.unique(y_train_final, return_counts = True)
#print(np.asarray((unique_elements, counts_elements)))

plt.bar( np.arange( 43 ), counts_elements, align='center',color='green' )
plt.xlabel('Class')
plt.ylabel('No of Training data')
plt.xlim([-1, 43])

#plt.show()

#normalize my data
for i in X_train_final:
    i = i*(1/225)




