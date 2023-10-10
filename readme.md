# Breast Cancer Prediction

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Data Processing](#data-processing)
- [Training the Model](#training-the-model)
- [Web Application](#web-application)
- [Integration of Model](#integration-of-model)
- [Deployment](#deployment)
- [Team Members](#team-members)
- [References](#references)
- [How to Run the Project](#how-to-run-the-project)

## Introduction
This repository contains the code and resources for a breast cancer prediction project. The project aims to leverage machine learning and convolutional neural networks (CNNs) for early detection of breast cancer using pathology scan images.

## Prerequisites
Before getting started, ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow
- Flask
- Heroku CLI

## Project Overview
The project involves the development of a machine learning model for breast cancer prediction and its integration into a user-friendly web application. The project is organized into various components, including data processing, model development, web application creation, and deployment.

## Technology Stack
- Google Colab
- Python with TensorFlow
- Flask
- Heroku

## Model Architecture
The project utilizes custom CNN models with 15 and 21 layers. These models are designed to analyze breast pathology scan images for cancer detection.

## Data Processing
We preprocess breast pathology scan images, including resizing and pixel normalization, to ensure compatibility with the model's input requirements.

## Training the Model
The machine learning model is trained using curated datasets. This involves training, testing, and validation sets to achieve high accuracy in cancer detection.

## Web Application
The user-friendly web application is built using the Flask framework, which allows users to upload pathology scan images for cancer prediction.

## Integration of Model
The trained model is seamlessly integrated into the web application, providing real-time cancer prediction based on user-uploaded images.

## Deployment
The web application is deployed on Heroku, making it accessible to users online. Heroku offers scalability and ensures the model's accessibility.

## Team Members
- Srinivas Reddy Chitukula (Developer, Deployment)
- Krishna Vipul Shah (Documentation, Testing)
- Amarnadh Kari (Project Coordinator)
- Bala Chikkala (Developer)

## References
- [Kaggle Dataset](https://www.kaggle.com/code/sukeshtech17/breast-cancer-end-to-end-system-cnn)
- [IEEE Xplore Research Paper](https://ieeexplore.ieee.org/document/7900002)
- [ResearchGate Research Paper](https://www.researchgate.net/publication/328728209_Breast_cancer_histology_images_classification_Training_from_scratch_or_transfer_learning)

## How to Run the Project
Follow these steps to run the breast cancer prediction project:

```bash
# Clone the repository
git clone https://github.com/nivasreddy99/AI_project.git

# Navigate to the project directory
cd AI_project

# Install the required Python packages
pip install -r requirements.txt

# Run the Flask application
python app.py
