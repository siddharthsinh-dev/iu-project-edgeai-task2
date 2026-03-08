
# Edge AI : Age, Gender and Expression Recognition Application

**Project: Edge AI (DLBAIPEAI01)**  
**Task 2: Age, Gender and Expression Recognition Application**

The project compares two different approaches for facial attribute recognition:

1. A cloud-based implementation using the DeepFace framework in Google Colab
2. A mobile edge AI implementation running directly on an iPhone using CoreML models

The purpose of the project is to observe the differences between cloud-based inference and on-device inference in terms of accuracy & other evaluation metrics and inference time.

---

# Repository Structure

edgeAI_ios_app/          -> iOS application project (Xcode)  
edgeAI_ios_app_results/  -> Results and evaluation plots of the iOS App and Working app screenshot
google_colab_notebook/                -> Google Colab notebook implementation  
google_colab_notebook_results/        -> Results and evaluation plots from Colab  


---

# Part 1 — Running the Model in Google Colab

The cloud implementation uses the DeepFace framework to analyse facial attributes from images.

## Step 1: Open Google Colab

Go to:

https://colab.research.google.com/

Upload the notebook file located in:

google_colab_notebook/

---

## Step 2: Install Required Libraries

Run the following commands in the first cell of the notebook:

!pip install deepface  
!pip install opencv-python  
!pip install matplotlib  

---

## Step 3: Import Required Libraries

The notebook uses the following Python libraries:

from deepface import DeepFace  
import cv2  
import os  
import time  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score  
from sklearn.metrics import recall_score  
from sklearn.metrics import f1_score  
from sklearn.metrics import confusion_matrix  

---

## Step 4: Upload the Evaluation Images

The experiment uses 20 manually selected images.

Upload them directly inside Colab using:

from google.colab import files  
uploaded = files.upload()

The file names follow the format:

*01_adult_man_happy.jpg*  
*02_elderly_woman_sad.jpg*

The naming format is used to manually assign the ground truth labels for evaluation.

---

## Step 5: Run DeepFace Analysis

The DeepFace framework automatically predicts:

Age (Adult / Elderly)  
Gender (Man / Woman)  
Facial Expression (Happy  / Sad)

Example usage:

DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)

The notebook loops through all uploaded images and stores predictions in a results table.

---

## Step 6: Evaluate Model Performance

The following evaluation metrics are computed:

Accuracy  
Precision  
Recall  
F1 Score  
Model Inference Time (ms)

Evaluation Results, Metrics summary and Confusion Matrices are generated to visualise the results.

These outputs are stored in the folder:

google_colab_notebook_results/

---

# Part 2 — Running the Edge AI iOS Application

The second implementation runs directly on the iPhone using CoreML models.

The following pretrained models are used:

AgeNet.mlmodel  
GenderNet.mlmodel  
CNNEmotions.mlmodel  

These models were obtained from:

https://github.com/cocoa-ai/FacesVisionDemo

---

# Requirements

To run the iOS application you need:

macOS  
Xcode (recommended: latest version)  
An iPhone device (used for testing)  
USB cable to connect the iPhone to the Mac  

---

# Running the iOS Application

## Step 1 — Open the Project

Open the following file in Xcode:

edgeAI_ios_app/EdgeAITask2.xcodeproj

---

## Step 2 — Select the Target Device

In the top toolbar of Xcode select your connected iPhone device instead of the simulator.

Example:

EdgeAITask2 → iPhone Name / Model

---

## Step 3 — Connect the iPhone

Connect the iPhone to your Mac using a USB cable.

If this is the first time running an app:

1. Allow developer mode on the iPhone from iPhone settings  
2. Trust the computer if prompted  

---

## Step 4 — Build the Project

Press:

Cmd + B (Build)

This will compile the project and ensure there are no build errors.

---

## Step 5 — Run the Application

Press:

Cmd + R (Run)

Xcode will install the app directly on the iPhone.

---

# Using the Application

After launching the application:

1. The app interface will ask the user to select photo from photo gallery 
2. The user clicks Analyze Face Button
3. The face in the photo will be detected
4. The application predicts:  
Age (Adult / Elderly)  
Gender (Man / Woman)  
Facial Expression (Happy  / Sad)  
Model Inference time is also displayed in (ms)

The predictions are displayed directly on the screen.

Screenshot of the application output are included in:   
edgeAI_ios_app_results/

---

# Note on Model Files

The CoreML model files are large.

They are stored in the repository using Git LFS (Large File Storage) so that the repository can handle files larger than the normal Git limit.

**If cloning the repository manually, Git LFS may need to be installed.**

---

# Author
Siddharthsinh Rathod   
Project: Edge AI (DLBAIPEAI01)  
IU International University of Applied Sciences
