# Regalscan
Scanner für Produkte aus Einkaufsliste im Supermarktregal / Python Prototyp

# Task
This demo script shall evaluate an idea of shopping assistance:
A list of often bought articles shall be detected and highlighted in the supermarket shelf in order to assist the user finding the articles and speed up the shopping experience

# Method
The task is divided into 3 steps:
- Extraction of the feauture vectors for each article
- Object detection of relevant object (a standard YOLO model is used)
- Feature vector calculation and correlation to all pre-scanned articles
Finally, if an object is detected, it is marked in the video

# Realization
The script is a python script using Pytorch, Ultralytics and Timm framework.
It can be run on a PC with GPU support.

performed steps:
Object detection on a standard YOLO 8 small model. The small model is absolutely sufficient as only the bounding boxes of relevant objects are used. The detected classes are not important. Classes are only used to delect irrelevant objects like persons.

Main evaluation step:
For all pre-selected bounding boxes, a feaure vector is being calculated
It has turned out, that the ConvNext Tiny model, pretrained in22ft1k achieves the best performance in generating unique feature vectors.
To further improve the differentition performance, the most relevant elements inside the feature vectors are being selected

# Prerequisits
Pytorch, Timm, Ultralytics, OpenCV shall be installed via PIP
My versions are:
...
(This is running safe on Win11 + I13700 + GTX4070Ti)

# Run
To run and evaluate start main.py

# Test files
In the example folder, there are demo pictures of objects to be detected and a video simulating the walk through a supermarket
Feel free to use other article photos and videos!

 
