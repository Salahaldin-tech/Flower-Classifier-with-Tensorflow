import tensorflow as tf
import numpy as np 
import json
from PIL import Image 

def process_image (image):
    # converting from numpy to tf
    tf_image=tf.convert_to_tensor(image)
    # resize the image 
    tf_image=tf.image.resize(tf_image,(224,224))
    #normalize it
    tf_image/=255.0

    
    return tf_image.numpy()


def predict(image_path,model,top_k):
    # opening the image from the path
    image=Image.open(image_path).convert('RGB') 
    
    image=np.asarray(image)
    
    #calling the process_image for resize and normalize
    image=process_image(image)
    
    # to make the shape as (1, 224, 224, 3)
    image=np.expand_dims(image,axis=0)
    
    #using the model to predict
    pred=model.predict(image)
    
    # getting the tip k probs and their index
    top_k_probab,top_k_index= tf.nn.top_k(pred[0], k=top_k)
    
    #converting them from tensor to numpy 
    top_k_probab=top_k_probab.numpy()
    top_k_index=top_k_index.numpy()

    
    
    top_classes=[]
    
    for index in top_k_index:
        top_classes.append(str(index))
        
    print(top_classes)
    return top_k_probab, top_classes

def get_class_map(json_file_path):
    with open(json_file_path, 'r') as f:
        class_names = json.load(f) 

    return class_names