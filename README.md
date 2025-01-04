# **Deep Learning for Image Classification in Computer Vision and Pathology**

## **Description**
This project focuses on leveraging **Convolutional Neural Networks (CNNs)** to tackle image classification tasks in **computer vision** and **computational pathology**. Through the use of real-world datasets, students will train and analyze CNN models to classify images across diverse applications. The project also explores **transfer learning**, applying pre-trained models from one domain to another, and evaluates their performance on unseen datasets.

### **Main Objectives**
1. Train and fine-tune CNN models to classify images from pathology and computer vision datasets.
2. Conduct detailed analyses of model performance, including **dimensionality reduction** and **feature visualization** using **t-SNE**.
3. Explore **transfer learning** by applying trained CNN encoders to datasets from different domains.
4. Utilize **classical machine learning techniques** to classify features extracted from the CNN encoders.

---

## **Table of Contents**
- [Datasets](#datasets)
- [Required Libraries](#required-libraries)
- [Running the Project](#running-the-project)
- [Running Tester Code](#running-tester-code)
- [Detailed Tasks](#detailed-tasks)
  - [Task 1: Training and Feature Analysis](#task-1-training-and-feature-analysis)
  - [Task 2: Transfer Learning and Feature Classification](#task-2-transfer-learning-and-feature-classification)
- [Learning Outcomes](#learning-outcomes)

---

## **Datasets**
This project uses three datasets, reduced to manageable sizes, representing pathology and computer vision applications:

### 1. **Colorectal Cancer Classification**
- **Original Dataset**: 100k image patches, 8 classes.
- **Project Dataset**: 6k image patches, 3 classes:
  - Smooth Muscle (MUS)
  - Normal Colon Mucosa (NORM)
  - Cancer-Associated Stroma (STR)

### 2. **Prostate Cancer Classification**
- **Original Dataset**: 120k image patches, 3 classes.
- **Project Dataset**: 6k image patches, 3 classes:
  - Prostate Cancer Tumor Tissue
  - Benign Glandular Prostate Tissue
  - Benign Non-Glandular Prostate Tissue

### 3. **Animal Faces Classification**
- **Original Dataset**: 16k images, 3 classes.
- **Project Dataset**: 6k images, 3 classes:
  - Cats
  - Dogs
  - Wildlife Animals

---
## **Required Libraries**

1. torch: Machine learning library
2. numpy: For math library and matrices
3. sklearn: For SVM model and classification reports
4. matplotlib: For graphing
5. torchvision: For machine learning models
6. google.colab: For access to Google Drive

---
## **Running the Project**
To run the project just download the .ipynb files names "COMP432_Task1.ipynb" for Task 1 and "COMP432_Task2.ipynb" for Task 2 and put into a Google Colab. Our datasets and models were saved individually to our own Google Drive accounts and therefore, it would be necessary for anyone running the code to have the same directories within their own Google Drive. As connecting a centralized database to our Colab proved difficult, or temporally infeasible with the shear size of the datasets which need to be fetched, where download speeds are variable to the users internet connection, we will offer an easy way to set up your own directory the way the Colab expects within your own drive. The directory heirarchy is:

<img width="248" alt="Screenshot 2024-11-27 at 12 34 09 PM" src="https://github.com/user-attachments/assets/4c4f7da1-a4ef-40f1-ac8a-d16dceb415f8">

Both the resnet18_model.pth and the resnet18_model_V1.pth are models trained by the team. Here is a link to a .zip that holds this directory, which you can just drag and drop into your drive (the program will then prompt you to mount your drive and you will be good to go): https://drive.google.com/file/d/1YC1mEr8vwjMOnX1axQttGDjsRcYAeHCG/view?usp=sharing

In the future, using a more centralized database such as Google Cloud Storage while not downloading but instead only fetching the individual datasets when needed will prove more efficient and collaborative.

---
## **Running Tester Code**
Attached to this Github is a program that is not one of the two tasks, but is instead used to give users the ability to try out a colorectal cancer test set on our ResNet18 model, called "TesterCode.ipynb". A sample dataset is available in .zip form with this link and just drag and drop the dataset into your MyDrive: https://drive.google.com/file/d/1zIVXde5GqLr2xW0jN6i5KKrj2vqiZrJZ/view?usp=sharing

Once in your drive, put the tester code notebook into Google Colab and once running will prompt you to mount your Drive account. The notebook will be expecting the path "/content/drive/MyDrive/TestDataset" so make sure that the directory you place in your main Drive is called "TestDataset". Within this directory should be -> "colorectal_cancer" -> MUS and STR and NORM which contain the images.

---
## **Detailed Tasks**

### **Task 1: Training and Feature Analysis**
- Train a **CNN model** (e.g., ResNet, VGG, AlexNet) on **Dataset 1 (Colorectal Cancer)**.
- Report **training accuracy and loss** for all experiments.
- Apply **t-SNE** to visualize output features from the CNN encoder according to class labels.
- Provide detailed discussions on the observed model behaviors.

---

### **Task 2: Transfer Learning and Feature Classification**

#### **1. Transfer Features Across Domains**
- Use the **trained CNN encoder** from Task 1 on **Dataset 2 (Prostate Cancer)** and **Dataset 3 (Animal Faces)**.
- Analyze and visualize extracted features using **t-SNE**.

#### **2. Compare with Pre-trained Models**
- Use a **pre-trained ImageNet CNN encoder** to extract features for **Dataset 2** and **Dataset 3**.
- Perform similar **t-SNE analyses** for comparison.

#### **3. Classify Extracted Features**
- Apply **classical machine learning algorithms** (e.g., SVM, Random Forest, Logistic Regression) to classify the features from **one encoder per dataset**.

---

## **Learning Outcomes**
This project aims to provide hands-on experience with:
- Training, tuning, and analyzing CNN models.
- Applying **transfer learning** across different domains.
- Visualizing and interpreting features using **t-SNE**.
- Integrating **deep learning** with **classical machine learning** techniques.

By exploring both pathology and computer vision datasets, students will gain insights into the versatility and challenges of applying deep learning techniques to diverse real-world applications.

