# Chicken-Disease-Classification--Project

Absolutely! Here's the entire content in a single snippet:

````markdown
# Chicken Disease Classification

## Overview

This repository contains a machine learning project for the classification of chicken diseases using image data. The goal is to develop a model that can accurately identify common diseases in chickens based on images of affected birds. This project can be useful in early detection and monitoring of diseases in poultry farms.

## Project Structure

- **data**: Contains the dataset used for training and testing the model.
- **src**: Contains the source code for the machine learning model.
  - **stage_01_data_ingestion.py**: Code for data preprocessing.
  - **stage_02_base_model_preparation.py**: Implementation of the machine learning model.
  - **stage_03_training.py**: Script for training the model.
  - **stage_04_evaluation.py**: Script for evaluating the model.
  - **predict.py**: Script for making predictions on new data.
- **logs**: Contains log files documenting the training process.

## Project Structure Details

Our project is thoughtfully organized to ensure clarity, maintainability, and effective collaboration. Here's a breakdown of the key components:

### `artifacts`

This directory is the repository for the dataset crucial to both training and testing our machine learning model. The quality and organization of the data within this directory are pivotal, directly influencing the model's accuracy and reliability. A well-prepared dataset is the foundation for the success of the entire project.

### `src`

The `pipeline`directory contains the source code for our machine learning model, meticulously organized into several stages:

#### `stage_01_data_ingestion.py`

This script handles data preprocessing, shaping the raw data to meet the specific requirements of our model. Effective data preprocessing ensures that the model is fed with high-quality, well-organized data, optimizing its learning process.

#### `stage_02_base_model_preparation.py`

In this script, we implement the core architecture of our machine learning model. The base model is designed to provide a robust foundation for subsequent training. A well-crafted base model is essential for achieving high performance and generalization on unseen data.

#### `stage_03_training.py`

This script is the heartbeat of our project, orchestrating the training process. It utilizes the prepared dataset to iteratively improve the model's performance. Training involves adjusting model parameters based on the provided data, allowing the model to learn and adapt to the nuances of our specific problem.

#### `stage_04_evaluation.py`

Dedicated to evaluating the model's performance, this script employs various metrics to assess its accuracy and effectiveness. The insights gained from the evaluation process guide us in refining the model and making informed decisions about potential adjustments or enhancements.

#### `predict.py`

This forward-thinking script enables us to make predictions on new, previously unseen data. It extends the utility of our model beyond the training dataset, allowing for real-world application and ensuring the model's practicality in diverse scenarios.

### `logs`

The `logs` directory is integral to our commitment to transparency and accountability. Here, we document the training process in detail. The log files capture crucial information such as training loss, accuracy, and other performance metrics. These logs serve not only as a historical record but also as a diagnostic tool, aiding in the continuous improvement and optimization of our machine learning model over time.

# How to run?

### STEPS:

Clone the repository

```bash
https://github.com/entbappy/Chicken-Disease-Classification--Project
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.8 -y
```

```bash
conda activate cnncls
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up you local host and port
```

### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

    #with specific access

    1. EC2 access : It is virtual machine

    2. ECR: Elastic Container registry to save your docker image in aws


    #Description: About the deployment

    1. Build docker image of the source code

    2. Push your docker image to ECR

    3. Launch Your EC2

    4. Pull Your image from ECR in EC2

    5. Lauch your docker image in EC2

    #Policy:

    1. AmazonEC2ContainerRegistryFullAccess

    2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image

    - Save the URI: 854337983934.dkr.ecr.eu-west-2.amazonaws.com/chicken

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:

    #optinal

    sudo apt-get update -y

    sudo apt-get upgrade

    #required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

# 6. Configure EC2 as self-hosted runner:

    setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

## Future Improvements

This project can be extended by:

- Exploring more advanced models and transfer learning techniques.
- Enhancing the dataset with additional images and classes.
- Fine-tuning hyperparameters for better performance.

Feel free to contribute, open issues, or provide feedback!
````
