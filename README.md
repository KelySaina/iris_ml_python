# Project : IRIS

## Context

The Iris dataset is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in his 1936 paper titled "The use of multiple measurements in taxonomic problems." It is sometimes called Anderson's Iris dataset because Edgar Anderson collected the data to quantify the morphological variation of Iris flowers of three related species.

The dataset consists of 50 samples from each of the three species of Iris (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured for each sample: the lengths and widths of the sepals and petals, in centimeters. This dataset has become a typical testing case for many statistical classification techniques in machine learning, such as support vector machines.

## Content

The dataset contains a total of 150 records spread across 5 attributes:
- Sepal length (sepal_length)
- Sepal width (sepal_width)
- Petal length (petal_length)
- Petal width (petal_width)
- Class (species)

The goal of this project is to predict the `species` column based on the other attributes.

## Project Structure

### 1. Data Loading

The data is loaded from a CSV file. The four measured features for each sample are used as independent variables to predict the Iris species.

### 2. Data Preparation

The data is cleaned and prepared for model training. This includes encoding the labels of the `species` column.

### 3. Model Training

We use several machine learning algorithms to train our models:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Linear Regression

### 4. Model Evaluation

The models are evaluated using performance metrics such as accuracy, recall, and F1 score.

### 5. Data Visualization

Visualizations are created to explore the data and the models' results. This includes swarm plots for each feature by species.

### 6. Streamlit Application

A Streamlit application is developed to enable interaction with the models and visualizations. The application includes the following sections:
- Sample data display
- Model explanation selection
- Data visualization
- User predictions

## Usage

To run the Streamlit application, follow these steps:

1. Clone the repository.
    ```sh
    git clone https://github.com/KelySaina/iris_ml_python
    ```
2. Install the necessary dependencies from the `requirements.txt` file.
3. Run the Streamlit application with the command:
    ```sh
    streamlit run py-scripts/app-streamlit.py
    ```

    Run the CLI application with the command:
    ```sh
    streamlit run py-scripts/app-cli.py
    ```

## Authors

This project was developed by KelySaina as part of a study on statistical classification techniques and machine learning.
