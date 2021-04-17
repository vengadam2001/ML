#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm

# Set batch size for training and validation
batch_size = 32


# In[ ]:


model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}


# In[ ]:


model_dictionary


# In[ ]:


img_height = 224
img_width = 224
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./archive/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./archive/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[ ]:


model_benchmarks = {'model_name': [], 'num_model_params': [], 'validation_accuracy': []}
for model_name, model in tqdm(model_dictionary.items()):
    # Special handling for "NASNetLarge" since it requires input images with size (331,331)
    if 'NASNetLarge' in model_name:
        input_shape=(331,331,3)
        train_processed = train_ds
        validation_processed = val_ds
        print("nasa")
    else:
        input_shape=(224,224,3)
        train_processed = train_ds
        validation_processed = val_ds
        print("others")
    base_model = model(include_top=False,input_shape=input_shape)
    base_model.trainable = False
    print("models loaded")
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(train_ds.class_names))
    model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(
input_shape[0], input_shape[1], interpolation='bilinear', name=None),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    base_model,
    global_average_layer,
    tf.keras.layers.Dense(10),
    # tf.keras.layers.Dense(10),
    prediction_layer
    ])
    # model.summary()
    base_learning_rate = 0.001
    model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    # We can evaluate the model right now to see how it does before training it on our new images
    initial_epochs = 1
    # validation_steps=20

    # loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
    history = model.fit( train_ds ,
    epochs=initial_epochs,
    shuffle=True,
    batch_size=32,
    validation_data=val_ds,
    )
    model_benchmarks['model_name'].append(model_name)
    model_benchmarks['num_model_params'].append(base_model.count_params())
    model_benchmarks['validation_accuracy'].append(history.history)
#     model_benchmarks['accuracy'].append(history.history['accuracy'][-1])


# In[31]:


len(train_ds.class_names)


# In[29]:


# load the pre-trained model with global average pooling as the last layer and freeze the model weights
   

   # custom modifications on top of pre-trained model
   clf_model = tf.keras.models.Sequential()
   clf_model.add(pre_trained_model)
   clf_model.add(tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax'))
   clf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
   history = clf_model.fit(train_processed, epochs=3, validation_data=validation_processed,steps_per_epoch = 15)

   # Calculate all relevant metrics
   model_benchmarks['model_name'].append(model_name)
   model_benchmarks['num_model_params'].append(pre_trained_model.count_params())
   model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])


# In[ ]:




