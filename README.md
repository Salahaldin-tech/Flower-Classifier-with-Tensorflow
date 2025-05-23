# Flower Classifier with TensorFlow🌸

> A deep learning project to classify flowers into 102 categories using transfer learning with TensorFlow Hub and MobileNet.

![Your paragraph text](https://github.com/user-attachments/assets/cf20a177-73e6-4113-a8b4-abeb5fba5a92)


---

## Project Overview :

This project classifies flower images using a pre-trained MobileNet model. It's built as part of the [Udacity AI Nanodegree](https://www.notion.so/%5B%3Chttps://www.udacity.com/course/intro-to-machine-learning-with-tensorflow--ud187%3E%5D(%3Chttps://emc.udacity.com/c/palestine-launchpad-google/catalog/An0jgdK3/i/nd/nd089-ent-google-si%3E)) and demonstrates key concepts of **image preprocessing**, **transfer learning**, and **model evaluation**.

---

## Features:

- 🔍 Uses **MobileNet v2** from TensorFlow Hub
- 🎓 Trained on the **Oxford Flowers 102** dataset
- 🛠️ Command-line prediction using `predict.py`
- 📊 Accuracy up to 90%
- 📁 Supports JSON-based category mapping

---

## Demo 🖼️

Here’s an example of the classifier in action:
After entering the command line :

be sure to pass the paths of image ,JSON file, number of classes needed , the trained model.

```python
$ python ./test_images/orchid.jpg my_model.h5 --category_names label_map.json --top_k 5
#which passes the json file path, top k classes the flower would be , the trained model:

```

Results for the command above:


![Screenshot 2025-05-23 214033](https://github.com/user-attachments/assets/65f1251e-8df0-4f7a-aa78-a79ef8b97e93)

![image](https://github.com/user-attachments/assets/096f3959-d90c-47b0-a1a9-37b452e01131)


---

project sturcute:

┣ [📜predict.py] # Command-line prediction tool

┣ [📜helpingfun.py] # Helper functions (image processing,predict function, loading model)

┣ 📜Project_Image_Classifier_Project.ipynb # Jupyter Notebook for training the model

┣ 📜my_model.h5 # Saved trained model

┣ 📜label_map.json # Flower category to name mapping

┣ 📁test_images/ # Sample images for testing

┣ 📁assets/ 

┗ 📜readme.md
---

## Model Training (Project_Image_Classifier_Project.ipynb)

The project is broken down into multiple steps:

- Load the image dataset and create a pipeline.
- Build and Train an image classifier on this dataset.
- Testing the Network
- Use your trained model to perform inference on flower images.
- Sanity Check with a good visualization for the results

---

## Helper Functions — `helpingfun.py`

This script contains reusable utility functions used by both the training notebook and the prediction script. It helps streamline core operations like:

- **`process_image(image_path)`**
    
    Preprocesses an image to the format expected by the model (resize, normalize).
    
- **`predict(image_path, model, top_k)`**
    
    Runs inference on a given image and returns the top K predictions with probabilities.
    
- **`load_class_names(json_path)`**
    
    Loads a `.json` file mapping category indices to flower names.
    

---

### `predict.py` – Image Classifier Script

This script is used to make predictions on new flower images using a trained model (`.h5`). It can be run from the command line and supports custom settings for:

- Image path
- Trained model file
- Top K predictions to display
- Mapping class indices to flower names

commands to use the model

### **Examples**

For the following examples, i assume that you have a file called `orchid.jpg` in a folder named`/test_images/` that contains the image of a flower. i also assume that we have a Keras model saved in a file named `my_model.h5`.

**Basic usage:**

```python
$ python predict.py ./test_images/orchid.jpg my_model.h5

```

**Options:**

- Return the top 3 most likely classes:

```python
$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3

```

- Use a `label_map.json` file to map labels to flower names:

```python
$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json

```


## Author

**Salahaldin-tech**  
[GitHub Profile](https://github.com/Salahaldin-tech)

