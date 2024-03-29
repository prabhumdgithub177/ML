# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras
# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution                   
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
           
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer, no need input size coz its already there.
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
# 0-255 is pixel values
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset2/trainingset',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset2/testset',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


from IPython.display import display
from PIL import Image
classifier.fit_generator(training_set,
                         steps_per_epoch = 2000, #8k
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 500)  #2k


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('image1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >=0.5:
    prediction = 'dog'
else:
    prediction = 'cat'   
print(prediction)



