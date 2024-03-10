from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

"""
This Python file is a container for our models, it simply works by holding the different 
Neural Network designs that we can train our model with.

NN function is the main neural network used for this model since it possesses a very well thought
design to deliver the answer to our problem.
"""
def NN(X_train):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=X_train.shape[1])) # input layer with 512 neurons
    model.add(Dropout(0.5)) # dropout layer 
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))) # hidden layer
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid')) # final layer with 1 neuron to get our binary result using sigmoid
    return model

def NN_activation_mixed(X_train):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(256, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))

    return model