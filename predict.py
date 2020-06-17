#Import libraries:

import tensorflow as tf
import tensorflow_hub as hub
import argparse 
from PIL import Image 
import numpy as np 
import json


parser = argparse.ArgumentParser (description = 'Image classification')
# parser.add_argument('image_path', help = 'Path of an image to predict')

parser.add_argument ('--image_path',default = './test_images/orange_dahlia.jpg', help = 'Path of the image file to predict', type = str)
parser.add_argument('--model', default = 'model.h5', help = 'Model path', type = str)
parser.add_argument ('--topk', default = 5, help = 'Top K most results', type = int)
parser.add_argument ('--classes' , default = 'label_map.json', help = 'Class names', type = str)
arg = parser.parse_args()

                     
image_path = arg.image_path
model = arg.model
topk = arg.topk
classes =  arg.classes
 
    
# Read and load json file:
                     
# with open(classes, 'r') as file:
#         class_names = json.load(file)  
# class_names_new = dict()
# for key in class_names:
#     class_names_new[str(int(key)-1)] = class_names[key]   
 

with open(arg.classes, 'r') as file:
    class_names = json.load(file)
# class_names_new = dict()
# for key in class_names:
#     class_names_new[str(int(key)-1)] = class_names[key]
    
    
    
model_load = tf.keras.models.load_model(model, custom_objects = {'KerasLayer' : hub.KerasLayer})
                     
                     
                     
#Image preprocessing:
def process_image(image):
    
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image /= 225
    image = image.numpy()
    
    return image     
                     
                     
class_names_new = dict()
for key in class_names:
    class_names_new[str(int(key)-1)] = class_names[key]
def predict(image_path, model_load, topk):
    
    #Importing
    image = Image.open(image_path)
    #Converting
    image = np.asarray(image)
    #Resizing and normalizing image
    image = process_image(image) 
    #Adding a dim
    image = np.expand_dims(image, axis = 0)
    pred = model_load.predict(image)
    pred = pred.tolist()
    #The top k
    probs, labels = tf.math.top_k(pred, k = topk)  
    #Converting to list
    probs = probs.numpy().tolist()[0]
    labels = labels.numpy().tolist()[0]

    flower_names = [class_names_new[str(name)] for name in labels]
    print('\n______________________________\n')                 
    print('\nFlower names: \n', flower_names)
    print('\nProbabilities: \n', probs)
    print('\n______________________________\n')                 


   
                     
if __name__ == "__main__":
    predict(image_path, model_load, topk)                