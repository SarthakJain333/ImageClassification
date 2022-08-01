# Importing necessary libraries
import os
import tensorflow as tf
from keras import datasets, layers, models
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd


# Creating a Model 
cnn = models.Sequential(
    [
     layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(360,360,3)),
     layers.MaxPooling2D((2,2)),
     layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
     layers.MaxPooling2D((2,2)),
     layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'),
     layers.MaxPooling2D((2,2)), 
     layers.Dropout(0.5),
     layers.Flatten(),
     layers.Dense(64,activation='relu'),
     layers.Dense(9,activation='softmax')
     ]
)

cnn.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

print(cnn.summary())

dir = '/content/train'

train_datagen = ImageDataGenerator(rescale = 1.0/255)
train_generator = train_datagen.flow_from_directory(
                                                    dir ,
                                                    batch_size = 20 ,
                                                    color_mode = 'rgb',
                                                  
                                                    class_mode = 'categorical',
                                                    target_size=(360,360),
                                                
                                                    )

validation_datagen = ImageDataGenerator(rescale = 1.0/255)
validation_generator = validation_datagen.flow_from_directory(
                                                              '/content/validation' , 
                                                              batch_size = 20 ,
                                                              class_mode='categorical',
                                                              color_mode = 'rgb',
                                                              target_size=(360,360))

testgen = ImageDataGenerator(rescale = 1.0/255)
test_gen = testgen.flow_from_directory(
    directory="/content/test_2",
    target_size=(360,360),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
    )
   
history = cnn.fit(
    train_generator,
    epochs = 5,
    validation_data=validation_generator
    )

pred=cnn.predict(test_gen,
                 verbose=1)

predicted_class_indices= np.argmax(pred,axis=1)

df = pd.read_csv('/content/sample_submission.csv')
df.head(5)

submission = pd.DataFrame({
    'Id':df['Id'],
    'Category': pred
})

submission.to_csv('submission_2.csv',index=False)



