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
from keras.api.models import Sequential
from keras.api.layers import Conv2D
from keras.api.layers import MaxPool2D
from keras.api.layers import Dense
from keras.api.layers import Flatten
from keras.api.optimizers import SGD

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

"""
all layers use ReLu (Rectified linear unit) activation function and the He weight initialization
ReLu is = max(0, x) so negative neurons are off or 0 and only the positive ones pass,
advantage is that it doesnt activate all the neurons at the same time.
He weight initialization method calc a random number using the Gaussian prob function, 
G with mean of 0 and standard deviation of sqrt(2/n) n is inputs to node.


Now we make the model, and first we have a feature extraction layer
with convolution and pooling layer and then a classifier backend 
that makes the prediction

to start we do a convolutional layer with kernal of 3,3 and filter(output) of 
32, followed by a max pooling layer (takes a kernal size and takes the max within that area)

then we can flatten the filter maps. 
then we can use a softmax activation function with 10 output. Between the 
flatten layer and output we can have an dense layer to interpret the features this case 100

softmax(softer version of argmax(gives the index of largest value))
"""
def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        #compiling model
        opt = SGD(learning_rate=.01, momentum=.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
