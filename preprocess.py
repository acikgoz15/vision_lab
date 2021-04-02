import numpy as np
import csv
import matplotlib.pyplot as plt
import os
training = 28709 + 1
testing = 3589
with open('fer2013/fer2013.csv', newline='') as csvfile:
     data = list(csv.reader(csvfile))

path = 'data'
os.mkdir(path)

train = np.ndarray(shape=(28709,1,48,48), dtype=np.float32)
test = np.ndarray(shape=(3589,1,48,48), dtype=np.float32)
test2 = np.ndarray(shape=(3589,1,48,48), dtype=np.float32)
label_train = np.ndarray(shape=(28709), dtype=int)
label_test = np.ndarray(shape=(3589), dtype=int)
label_test2 = np.ndarray(shape=(3589), dtype=int)

for i in range (1, training):
    image = [int(j) for j in data[i][1].split()]
    image = np.asarray(image)
    image = image.reshape(1, 48,48)
    image = image / 255 #Normalization
    train[i-1,:,:,:] = image
    label = np.zeros(7)
    label[int(data[i][0])] = 1
    label = int(data[i][0])
    label_train[i-1] = label


location = "{}/train_images.npy".format(path)
np.save(location, train)
location = "{}/train_labels.npy".format(path)
np.save(location,label_train)

#Test I
for i in range (training, training+testing):
    image = [int(j) for j in data[i][1].split()]
    image = np.asarray(image)
    image = image.reshape(1, 48,48)
    image = image / 255 #Normalization
    test[i-training,:,:,:] = image
    label = np.zeros(7)
    label[int(data[i][0])] = 1
    label = int(data[i][0])
    label_test[i-training] = label


location =  "{}/valid_images.npy".format(path)
np.save(location, test)
location = "{}/valid_labels.npy".format(path)
np.save(location,label_test)


#Test II
for i in range (training+testing, training+2*testing):
    image = [int(j) for j in data[i][1].split()]
    image = np.asarray(image)
    image = image.reshape(1, 48,48)
    image = image / 255 #Normalization
    test2[i-training-testing,:,:,:] = image
    label = np.zeros(7)
    label[int(data[i][0])] = 1
    label = int(data[i][0])
    label_test2[i-training-testing] = label

location =  "{}/test_images.npy".format(path)
np.save(location, test2)
location =  "{}/test_labels.npy".format(path)
np.save(location,label_test2)

#print(len(image))
#image = np.asarray(image)
#image = image.reshape(48,48)
#print(image.shape)
#image.astype(np.uint8)
#plt.imshow(image, cmap='gray')
#plt.show()
#images = np.genfromtxt('fer2013.csv', delimiter=',', skip_header=1)
#print(images.shape)
#image = images[0][1].split()
#print(images['b'][0])
#class_score = images[:,0]
#image = images[:,1]
#data_type = images[:,2]
#print(images[1])
#print(type(class_score))
#print(type(image))
#print(type(data_type))
#print(class_score.shape)
#print(image[1])
#print(data_type.shape)
#print(images[:,1].shape)
#print(type(images[0][0]))
#print(type(images[0][1]))
#print(type(images[0][2]))
#image = images[1][1]
#image = images[1][1].reshape(48, 48)
#print(image.shape)
#print(len(data))
#print(data[0])
#print(data[1][0])
#print("------")
#print(data[1][1])
#print("------")
#print(data[1][2])
#
#image = data[1][1]
#print(len(image))
#image = np.asarray(image)
#print(image.shape)

"""
image.reshape(48, 48)
image = image.astype(np.uint8)

print(image)
"""
