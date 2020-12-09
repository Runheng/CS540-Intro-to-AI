import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_dataset(training=True):
    '''
    Input: an optional boolean argument (default value is True for training dataset)
    Return: two NumPy arrays for the train_images and train_labels
    '''
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training == True:
        return (train_images, train_labels)
    return (test_images, test_labels)

def print_stats(train_images, train_labels):
    '''
    This function will print several statistics about the data.
    Input: the dataset and labels produced by the previous function; 
    does not return anything
    '''
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    
    # Total number of images in the given dataset
    num_images = len(train_images)
    print(num_images)
    
    # Image dimension
    x = np.shape(train_images[0])[0]
    y = np.shape(train_images[0])[1]
    print('{}x{}'.format(x,y))
    
    # Number of images corresponding to each of the class labels
    num_label = np.zeros(10)
    for i in range(num_images):
        num_label[train_labels[i]] += 1
    for i in range(10):
        print('{}. {} - {}'.format(i,class_names[i],int(num_label[i])))
        
    pass

def build_model():
    '''takes no arguments and returns an untrained neural network model'''
    # create a model, add layers in order
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape = (28,28)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))

    # compile using following params
    opt = keras.optimizers.SGD(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # m = [keras.metrics.Accuracy()]
    m = ['accuracy']
    model.compile(optimizer=opt, loss=loss_fn, metrics=m)
    return model

def train_model(model, train_images, train_labels, T):
    '''
    takes the model produced by the previous function and the dataset and labels produced
    by the first function and trains the data for T epochs; does not return anything
    '''
    model.fit(train_images, train_labels, epochs=T)
    pass

def evaluate_model(model, test_images, test_labels, show_loss=True):
    '''
    takes the trained model produced by the previous function and the test image/labels
    prints the evaluation statistics as described below (displaying the loss metric value
    if and only if the optional parameter has not been set to False)
    does not return anything
    '''
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    if show_loss:
        print('Loss:', '{:.4f}'.format(test_loss))
    accu = '{:.2f}'.format(test_accuracy * 100)
    print('Accuracy: {}%'.format(accu))
    pass

def predict_label(model, test_images, index):
    ''' 
    takes the trained model and test images, and prints the top 3 most likely labels
    for the image at the given index, along with their probabilities; 
    does not return anything
    '''
    result = model.predict(test_images)
    # get labels for the image at given index
    labels = result[index]
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    # print the top three most likely labels
    ret = []
    for i in range(10):
        ret.append([labels[i],class_names[i]])
    ret = sorted(ret)
    for i in range(3):
        print('{}: {}%'.format(ret[9-i][1],'{:.2f}'.format(100*ret[9-i][0])))
    pass

'''
(train_images, train_labels) = get_dataset()
print(type(train_images))
print(type(train_labels))
print(type(train_labels[0]))
(test_images, test_labels) = get_dataset(False)
print_stats(train_images, train_labels)
model=build_model()
print(model)
print(model.loss)
print(model.optimizer)
train_model(model, train_images, train_labels, 10)
evaluate_model(model, test_images, test_labels, show_loss=False)
evaluate_model(model, test_images, test_labels, show_loss=False)
evaluate_model(model, test_images, test_labels)
model.add(layers.Activation('softmax'))
predict_label(model, test_images, 1)
'''