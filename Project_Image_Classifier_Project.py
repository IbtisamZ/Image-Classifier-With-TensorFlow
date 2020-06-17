#!/usr/bin/env python
# coding: utf-8

# # Your First AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Oxford of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load the image dataset and create a pipeline.
# * Build and Train an image classifier on this dataset.
# * Use your trained model to perform inference on flower images.
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

# ## Import Resources

# In[3]:


# The new version of dataset is only available in the tfds-nightly package.
get_ipython().run_line_magic('pip', '--no-cache-dir install tfds-nightly --user')
# DON'T MISS TO RESTART THE KERNEL


# In[1]:


# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


# In[2]:


# TODO: Make all other necessary imports.

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt

import numpy as np
import time
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import glob



import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


# ## Load the Dataset
# 
# Here you'll use `tensorflow_datasets` to load the [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102). This dataset has 3 splits: `'train'`, `'test'`, and `'validation'`.  You'll also need to make sure the training data is normalized and resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet, but you'll still need to normalize and resize the images to the appropriate size.

# In[3]:


# Download data to default local directory "~/tensorflow_datasets"
get_ipython().system('python -m tensorflow_datasets.scripts.download_and_prepare --register_checksums=True --datasets=oxford_flowers102')


# In[4]:



# TODO: Load the dataset with TensorFlow Datasets. Hint: use tfds.load()

dataset , dataset_info = tfds.load('oxford_flowers102' ,as_supervised = True, with_info = True)

# TODO: Create a training set, a validation set and a test set.

training_set = dataset['test'] # Since the number of the training set should be larger.
validation_set = dataset['validation']
test_set = dataset['train']


# ## Explore the Dataset

# In[5]:


dataset_info


# In[6]:


# TODO: Get the number of examples in each set from the dataset info.

taining_examples = dataset_info.splits['test'].num_examples
validation_examples = dataset_info.splits['validation'].num_examples
test_examples = dataset_info.splits['train'].num_examples

print('The dataset has {:,} images in the training set.'.format(taining_examples))
print('The dataset has {:,} images in the validation set.'.format(validation_examples))
print('The dataset has {:,} images in the testing set.'.format(test_examples))


# TODO: Get the number of classes in the dataset from the dataset info.

num_of_classes = dataset_info.features['label'].num_classes
print('The dataset has {:,} classes.'.format(num_of_classes))


# In[7]:


# TODO: Print the shape and corresponding label of 3 images in the training set.

for image, label in training_set.take(3):
    image = image.numpy()
    label = label.numpy()
  
    plt.imshow(image)
    plt.show()
    
    print('The shape of this image is:', image.shape)
    print('The label of this image is:', label)
    print('\n\n-----------------------\n\n')


# In[8]:


# TODO: Plot 1 image from the training set. 

for image,label in training_set.take(1):
    image = image.numpy()
    label = label.numpy()
    
plt.imshow(image)
plt.show()

# Set the title of the plot to the corresponding image label. 
print("The label of this image is: ", label)


# ### Label Mapping
# 
# You'll also need to load in a mapping from label to category name. You can find this in the file `label_map.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/3.7/library/json.html). This will give you a dictionary mapping the integer coded labels to the actual names of the flowers.

# In[9]:


with open('label_map.json', 'r') as f:
    class_names = json.load(f)


# In[14]:


# TODO: Plot 1 image from the training set. Set the title 
# of the plot to the corresponding class name. 

for image,label in training_set.take(1):
    image = image.numpy()
    label = label.numpy()
    
    plt.imshow(image)
    plt.show()

    print('The class name of this image is: ', class_names[str(label)])


# ## Create Pipeline

# In[18]:


# TODO: Create a pipeline for each set.

batch_size = 32
image_size = 224

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

# Data splitting:

training_batches = training_set.shuffle(taining_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)


# # Build and Train the Classifier
# 
# Now that the data is ready, it's time to build and train the classifier. You should use the MobileNet pre-trained model from TensorFlow Hub to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! 
# 
# Refer to the rubric for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load the MobileNet pre-trained network from TensorFlow Hub.
# * Define a new, untrained feed-forward network as a classifier.
# * Train the classifier.
# * Plot the loss and accuracy values achieved during training for the training and validation set.
# * Save your trained model as a Keras model. 
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right.
# 
# **Note for Workspace users:** One important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module. Also, If your model is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[28]:


# TODO: Build and train your network.

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape = (image_size, image_size,3))
feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(265, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(102, activation = 'softmax')

])
 

model.summary()


# In[29]:


model.compile(optimizer = 'adam', 
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    
save_best = tf.keras.callbacks.ModelCheckpoint('./best_model.h5',
                                               monitor='val_loss',
                                               save_best_only=True) 


history = model.fit(training_batches,
                    epochs = 20,
                    validation_data = validation_batches,
                    callbacks = [early_stopping])


# In[33]:


# TODO: Plot the loss and accuracy values achieved during training for the training and validation set.


training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Accuracy
plt.figure(figsize = (6, 6))
plt.plot(training_accuracy, label = 'Training Accuracy')
plt.plot(validation_accuracy, label = 'Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Model Accuracy')
plt.show()
# Loss
plt.figure(figsize = (6, 6))
plt.plot(training_loss, label = 'Training Loss')
plt.plot(validation_loss, label = 'Validation Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.show()


# ## Testing your Network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[34]:


# TODO: Print the loss and accuracy values achieved on the entire test set.

loss, accuracy = model.evaluate(testing_batches)

print('\n')
print('Accuracy after training: {:.4%}'.format(accuracy))
print('Loss after training: {:,.4f}'.format(accuracy))


# ## Save the Model
# 
# Now that your network is trained, save the model so you can load it later for making inference. In the cell below save your model as a Keras model (*i.e.* save it as an HDF5 file).

# In[36]:


# TODO: Save your trained model as a Keras model.

saved_keras_model_filepath = 'model.h5'
model.save(saved_keras_model_filepath)


# ## Load the Keras Model
# 
# Load the Keras model you saved above.

# In[37]:


# TODO: Load the Keras model

model = tf.keras.models.load_model('model.h5', custom_objects = {'KerasLayer':hub.KerasLayer})
model.summary()


# # Inference for Classification
# 
# Now you'll write a function that uses your trained network for inference. Write a function called `predict` that takes an image, a model, and then returns the top $K$ most likely class labels along with the probabilities. The function call should look like: 
# 
# ```python
# probs, classes = predict(image_path, model, top_k)
# ```
# 
# If `top_k=5` the output of the `predict` function should be something like this:
# 
# ```python
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# Your `predict` function should use `PIL` to load the image from the given `image_path`. You can use the [Image.open](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.open) function to load the images. The `Image.open()` function returns an `Image` object. You can convert this `Image` object to a NumPy array by using the `np.asarray()` function.
# 
# The `predict` function will also need to handle pre-processing the input image such that it can be used by your model. We recommend you write a separate function called `process_image` that performs the pre-processing. You can then call the `process_image` function from the `predict` function. 
# 
# ### Image Pre-processing
# 
# The `process_image` function should take in an image (in the form of a NumPy array) and return an image in the form of a NumPy array with shape `(224, 224, 3)`.
# 
# First, you should convert your image into a TensorFlow Tensor and then resize it to the appropriate size using `tf.image.resize`.
# 
# Second, the pixel values of the input images are typically encoded as integers in the range 0-255, but the model expects the pixel values to be floats in the range 0-1. Therefore, you'll also need to normalize the pixel values. 
# 
# Finally, convert your image back to a NumPy array using the `.numpy()` method.

# In[38]:


# TODO: Create the process_image function

def process_image(image):
    
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image /= 225
    image = image.numpy()
    
    
    return image
    


# To check your `process_image` function we have provided 4 images in the `./test_images/` folder:
# 
# * cautleya_spicata.jpg
# * hard-leaved_pocket_orchid.jpg
# * orange_dahlia.jpg
# * wild_pansy.jpg
# 
# The code below loads one of the above images using `PIL` and plots the original image alongside the image produced by your `process_image` function. If your `process_image` function works, the plotted image should be the correct size. 

# In[39]:


from PIL import Image

image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()


# Once you can get images in the correct format, it's time to write the `predict` function for making inference with your model.
# 
# ### Inference
# 
# Remember, the `predict` function should take an image, a model, and then returns the top $K$ most likely class labels along with the probabilities. The function call should look like: 
# 
# ```python
# probs, classes = predict(image_path, model, top_k)
# ```
# 
# If `top_k=5` the output of the `predict` function should be something like this:
# 
# ```python
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# Your `predict` function should use `PIL` to load the image from the given `image_path`. You can use the [Image.open](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.open) function to load the images. The `Image.open()` function returns an `Image` object. You can convert this `Image` object to a NumPy array by using the `np.asarray()` function.
# 
# **Note:** The image returned by the `process_image` function is a NumPy array with shape `(224, 224, 3)` but the model expects the input images to be of shape `(1, 224, 224, 3)`. This extra dimension represents the batch size. We suggest you use the `np.expand_dims()` function to add the extra dimension. 

# In[315]:


# TODO: Create the predict function

def predict(image_path, model, topk = 5):
    
    #Importing
    image = Image.open(image_path)
    
    #Converting
    image = np.asarray(image)
    
    #Resizing and normalizing image
    image = process_image(image)
    
    
    image = np.expand_dims(image, axis = 0)
    pred = model.predict(image)
    pred = pred.tolist()
    print(pred)
    
    
    #The top k
    probs, classes = tf.math.top_k(pred, k = topk)
    
    #Converting to list
    probs = probs.numpy().tolist()[0]
    print('\nProbs: ', probs)
    classes = classes.numpy().tolist()[0]
    print('\nClasses: ', classes)

    return probs, classes


# # Sanity Check
# 
# It's always good to check the predictions made by your model to make sure they are correct. To check your predictions we have provided 4 images in the `./test_images/` folder:
# 
# * cautleya_spicata.jpg
# * hard-leaved_pocket_orchid.jpg
# * orange_dahlia.jpg
# * wild_pansy.jpg
# 
# In the cell below use `matplotlib` to plot the input image alongside the probabilities for the top 5 classes predicted by your model. Plot the probabilities as a bar graph. The plot should look like this:
# 
# <img src='assets/inference_example.png' width=600px>
# 
# You can convert from the class integer labels to actual flower names using `class_names`. 

# In[335]:


i1 = './test_images/orange_dahlia.jpg'
i2 = './test_images/cautleya_spicata.jpg'
i3 = './test_images/wild_pansy.jpg'
i4 = './test_images/hard-leaved_pocket_orchid.jpg'


# In[336]:


class_names_new = dict()
for key in class_names:
    class_names_new[str(int(key)-1)] = class_names[key]


# In[339]:


def plotting(path):
   
    plt.figure(figsize=(8,4))
    
    
    img = Image.open(path)
    test_image = np.asarray(img)
    img = process_image(test_image)
    
    
    plt.subplot(1,2,1)
    title = path.rsplit("/",1)[-1]
    plt.title(title)
    plt.imshow(img)
    
 
    probs, labels = predict(path, model, 5)
    print(probs)
    print(labels)
    
    flower_names = [class_names_new[str(name)] for name in labels]
    print(flower_names)


    ax=plt.subplot(1,2,2)
    ax.yaxis.tick_right()
    plt.barh(flower_names,probs)


# In[340]:


plotting('./test_images/orange_dahlia.jpg')
plotting('./test_images/cautleya_spicata.jpg')
plotting('./test_images/wild_pansy.jpg')
plotting('./test_images/hard-leaved_pocket_orchid.jpg')


# In[341]:


get_ipython().system('jupyter nbconvert *.ipynb')


# In[ ]:




