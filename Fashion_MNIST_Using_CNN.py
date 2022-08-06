# coding: utf-8

#Importing Required Packages Tensorflow as Keras
import tensorflow as tf
from tensorflow import keras

#We will be using the built-in Fashion MNIST database - A pixel representation of 10 different fashion objects like(Shoes, Handbag, Shirt, #etc.)
fashion_mnist = tf.keras.datasets.fashion_mnist

#Let us load the data into train and test sets
(train_image, train_label), (test_image, test_label) = fashion_mnist.load_data()

# We will scale all the values to be between 0-1 inorder to make our CNN more effective. The highest pixel value is 255, so let us divide all #the pixel values with 255.0
train_image, test_image = train_image/255.0, test_image/255.0

# Our Train set has 60000 samples and test has 10000 samples. So, let us reshape them to pass through our CNN layer.
train_image = train_image.reshape(60000, 28, 28, 1)
test_image = test_image.reshape(10000, 28, 28, 1)



#Create a callback to stop training as soon as the validation accuracy reaches 90%
class myCallbac(tf.keras.callbacks.Callback):
    ''' This function will be executed at the end of the epoch when a desired accuracy is reached'''
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.90):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True


#Now lets build the network model 

model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                            tf.keras.layers.MaxPool2D(2, 2),
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                            tf.keras.layers.MaxPool2D(2, 2),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(10, activation='softmax')])


# The below line will print the summary of the network with output structure after each layer and the trainable parameters 

print(model.summary())


# Now, we need to compile the model by prom=viding the optimizer, in this case we use 'ADAM', loss function, here it's #'sparse_categorical_crossentropy' and finally the metrics we are going to evaluate the network's performance on: Accuracy

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Create an instance of the Callback to be used in our model

callback = myCallbac()


history = model.fit(train_image, train_label, epochs=10, callbacks=[callback])

print(f"Target accuracy of {round(history.history['acc'][-1]*100,2)}% reached at epoch: {history.epoch[-1]}")