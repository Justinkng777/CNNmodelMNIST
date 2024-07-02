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
from numpy import mean, std
from sklearn.model_selection import KFold

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

"""
To evaluate we use n-fold cross-validation in this case n = 5
so five fold cross validation
where each test set will be 20% of the training dataset.

it will be shuffled and then split into the goups.
n-fold cross validation means that it breaks the data set into a training and validation group
where we split the data into n groups, and train the model on n-1 of the group, and validate on the last group
we do this in a total of n times with the validation group being every group at least once

in this tutorial we are using the base epochs of 10 training, and a 
default patch size of 32 examples
"""

def evaluate_model(dataX, dataY, n_folds = 5):
    #makes variables into a list
    scores, histories = list(), list()

    #prepares cross validation
    kfold = KFold(n_folds, shuffle = True, random_state = 1)

    #enumerate splits for training and testing
    for train_ix, test_ix in kfold.split(dataX):

        #defining the model
        model = define_model()

        #where we select the rows for training and testing
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

        #where we fit model or train the model
        #training data, trainX, validation data trainY
        #returns a record of loss values and metric values during training, and then evaluation or accuracy
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        """
        training will train the model
        validation will see how accurate the model is
        holdout set or testing data is final estimate of performance
        """

        print('> %.3f' % (acc * 100.0))

        scores.append(acc)
        histories.append(history)
    return scores, histories


"""
we will create a diagnostic line plot that shows model performance
on the train and test set during each fold of k-fold cross validation.
With these graphs we can have an idea of if the model is overfitting, underfitting
or good fit for dataset

Blue lines = model performance on training dataset
orange lines= performance on holdout test dataset
"""
def summarize_diagnostics(histories):
     for i in range(len(histories)):
          #plot loss
          plt.subplot(2, 1, 1)
          plt.title('Cross Entropy Loss')
          plt.plot(histories[i].history['loss'], color='blue', label='train')
          plt.plot(histories[i].history['val_loss'], color='orange', label='test')

          #plot accuracy
          plt.subplot(2, 1, 2)
          plt.title('Classification Accuracy')
          plt.plot(histories[i].history['accuracy'], color='blue', label='train')
          plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
          plt.show()

"""
now classification accuracy scores in each fold being summarized
by calc mean and standard deviation
display distribution of scores by a box and whisker plot
"""
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    plt.boxplot(scores)
    plt.show()

#to actually test and run the model we have a testing harness
def run_test_harness():
     trainX, trainY, testX, testY = load_dataset()

     trainX, testX = prep_pixels(trainX, testX)

     scores, histories = evaluate_model(trainX, trainY)

     summarize_diagnostics(histories)

     summarize_diagnostics(scores)

run_test_harness()