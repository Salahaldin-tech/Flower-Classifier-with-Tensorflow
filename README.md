# 🌸 Flower Classifier with TensorFlow

> A deep learning project to classify flowers into 102 categories using transfer learning with TensorFlow Hub and MobileNet.

![Flower Classification](assets/flower-banner.gif)

---

## 📚 Project Overview

This project classifies flower images using a pre-trained MobileNet model. It's built as part of the [Udacity AI Nanodegree](https://www.udacity.com/course/intro-to-machine-learning-with-tensorflow--ud187) and demonstrates key concepts of **image preprocessing**, **transfer learning**, and **model evaluation**.

---

## 🚀 Features

- 🔍 Uses **MobileNet v2** from TensorFlow Hub
- 🎓 Trained on the **Oxford Flowers 102** dataset
- 🛠️ Command-line prediction using `predict.py`
- 📊 Accuracy up to **XX%** (add your number)
- 📁 Supports JSON-based category mapping

---

## 🖼️ Demo

Here’s an example of the classifier in action:
After entering the command line :
$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json --top_k 5
which passes the json file path, top k classes the flower would be , the trained model:
![image](https://github.com/user-attachments/assets/8ecc39b7-e62e-4ba3-a747-2d261941aad1)

![132png](https://github.com/user-attachments/assets/1aa1892f-2819-4f91-8a66-203bdc0431b3)

 



