"""
By justin Ng
Following tutorial by Machine learning mastery's article 
"How to Develop a CNN for MNIST Handwritten Digit Classification" by Jason Brownlee

Split the MNIST data set into training and testing
we will have to 
Develope baseline model
improve it 
and finilize and make predictions
"""
from keras.api.datasets import mnist
from matplotlib import pyplot as plt
from keras.api.utils import to_categorical

"""
#shows the first 9 data in the mnist set
(trainX, trainY), (testX, testY) = mnist.load_data()
#prints the shape, how many data and also dimensions
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
#ploting few first images
for i in range(9):
    #subplot 
    plt.subplot(330 + 1 + i)
    #plot raw pixel data
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
plt.show()
"""
#Load dataset, theyre all prealigned and 28x28 grayscale
def load_dataset():
    #loading in dataset using load_dataset function
    (trainX, trainY), (testX, testY) = mnist.load_data()

    #reshape data arrays to have single color channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    #one hot encode target values
    # have the numbers mapped to a binary vector
    """
    ex aa,ab,cd,aa would map to
    1,0,0 | 0,1,0 | 0,0,1 | 0,0,0
    only one is 'on'
    """
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY

#we need to prep the pixels to be used
def prep_pixels(train, test):
    #normalize the pixel values by converting to float and divide by 255
    # black being 0 and white being 255
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0

    return train_norm, test_norm