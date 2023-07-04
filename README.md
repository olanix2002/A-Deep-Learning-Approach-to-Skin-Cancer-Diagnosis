# A-Deep-Learning-Approach-to-Skin-Cancer-Diagnosis

This project aims to detect three types of skin cancer, namely basal cell carcinoma, melanoma, and squamous cell carcinoma using a deep learning-based approach.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [GitHub Workflow Development Process](#gitHub-Workflow-development-process)
- [Future Work](#future-work)
- [License](#license)

## Introduction

Skin cancer is one of the most common forms of cancer, and early detection is crucial for successful treatment. This project focuses on the detection of three types of skin cancer, namely basal cell carcinoma, melanoma, and squamous cell carcinoma, using a deep learning-based approach.

The model used in this project is a convolutional neural network (CNN), which is trained on a dataset of skin images. The CNN can take an image of a skin lesion as input and output the probability of the lesion being one of the three types of skin cancer.

## Installation
To install the required packages, please use the following command: pip install -r requirements.txt

## Dataset
The dataset used in this project is a collection of dermoscopic images of skin lesions, which have been labeled as one of the three types of skin cancer.
The dataset was preprocessed and split into training, validation, and test sets. The images were resized and normalized to improve the performance of the model.
The dataset used in this project is small, and the model's accuracy is affected due to a lack of data. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).

## Model Architecture
The model architecture used in this project is a CNN with the following layers:

- Input layer
- Data preprocessing layer (resizing and rescaling)
- Convolutional layers
- Max-pooling layers
- Flatten layer
- Fully connected layers

## Training
To train the model, follow these steps:

Download the dataset using the instructions mentioned in the Dataset section.

Run the following command: python train.py --data_path <path-to-dataset>
-
The trained model will be saved in the models directory.

## Evaluation
To evaluate the model, follow these steps:

Download the dataset using the instructions mentioned in the Dataset section.

Run the following command: python evaluate.py --data_path <path-to-dataset>

The evaluation results will be printed on the console.

## Results
The model achieved an accuracy of 85% on the test set, which is not optimal due to the small size of the dataset.

## Deployment
The skin cancer detection system was deployed using a FastAPI web application, which allows users to upload images of skin lesions and get a diagnosis. The model was packaged as a Python module and loaded into the FastAPI server. The server returns the accuracy and confidence level of the prediction. The code was committed to GitHub, and an Azure web app was created to deploy the model. The model was deployed successfully, and the API is accessible from a public URL.

## GitHub Workflow Development Process
The development process for this project was streamlined using the Azure GitHub workflow. The workflow consists of multiple steps, including linting, testing, building the application, and deploying the application to Azure. The workflow has been designed to automatically run on every commit to the repository, ensuring that the application is always up-to-date and functioning correctly.

To run the GitHub workflow, follow these steps:
- Clone the repository to your local machine
- Install the required dependencies by running the command pip install -r requirements.txt
- Create a new virtual environment by running the command python -m venv env
- Activate the virtual environment by running the command source env/bin/activate
- Run the GitHub workflow by running the command python .github/workflows/main.yml

The GitHub workflow will automatically run, and you can view the results in the GitHub Actions tab.

## Future Work
There are several areas where this project can be extended in the future:

- Use more advanced architectures for improved accuracy.
- Train the model on larger datasets to improve performance.
- Develop a user-friendly interface for the system.

## License
This project is licensed under the Apache License - see the LICENSE file for details.

