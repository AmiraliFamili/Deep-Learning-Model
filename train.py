from data import data_process
from models import NN, NN_activation_mixed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from keras.optimizers import Adam
from vis import plot_loss, plot_confusion_matrix
"""
This file is where we train our model we choose the neural network we want to use, number of epochs, batch size, verbose
, validation split they are all done here, this file can be considered the control room of the model, 

"""


# choose the path of the dataset
data_path = "CCD.xls"

# get the cleaned data using the data file and data_process function.
X_train, X_test, y_train, y_test = data_process(data_path)


# choose the neural network design you want to train with 

#model = NN(X_train)
model = NN_activation_mixed(X_train)

# choose optimizer (and learning rate)
optimizer = Adam(learning_rate=0.00001)

# compile the model with binary cross entropy loss function and accuracy for our metrics.
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# fit the model with the desired number of epochs, batch size, validation split and verbose.
history = model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.2, verbose=1)


#-------------- Model Evaluation -------------------------------


# predictions for the model
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# we take the predictions and use them to obtain the accuracy score
# or
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) # calculate the accuracy of the model 
print("Accuracy:", accuracy) 

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, classes=["No Default", "Default"])


# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)

# plot loss
plot_loss(history)
