# necessarily imports
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image

from helpingfun import process_image,predict,get_class_map

import json
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(prog='Flowers classifier',description='an application that can predict a type of flower using a flowers classifier trained ai model')
    parser.add_argument('image_path', help="Input image file path", type=str)
    parser.add_argument('model_path', help="Saved model path(.h5 extention)", type=str)
    parser.add_argument('--top_k', help="Top k probabilties", default=3,type=int)
    parser.add_argument('--category_names', help="Path to a json file mapping labels to flower names",type=str)
    
    return parser.parse_args()
    


def main():

    args=get_arguments()

    #loading the model
    loaded_model=tf.keras.models.load_model(args.model_path,custom_objects={"KerasLayer": hub.KerasLayer})
    #calling the predict function
    probs ,classes= predict(args.image_path, loaded_model,args.top_k)

    class_names=[]

    # we have to handle the error if the user didnt give a json file
    if args.category_names:
        classes_map=get_class_map(args.category_names)
        
        for i in classes:
             class_names.append(classes_map.get(str(i)))
    else:
        
        for _ in classes:
            class_names.append("Unknown flower")
    

   #printing results
    for flower_name , probability in zip(class_names,probs):
        print(f"{flower_name} --> the probability= {probability*100:.3f}%\n")

if __name__ == '__main__':
     main()
    

    





