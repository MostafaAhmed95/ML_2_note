import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2 as cv
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
