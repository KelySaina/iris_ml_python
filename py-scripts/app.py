import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from CSV
data = pd.read_csv('data/iris.csv')

# Ensure column names match expected format
data.columns = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'species']

# Encode species if necessary (assuming the species are strings)
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

# Prepare the data
X = data.iloc[:, :-1]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Streamlit app
st.title("Iris Species Prediction")

# Dataset display
st.write("### Iris Dataset")
st.write(data.head())

# Model selection
model_name = st.selectbox("Choose a model", [
                          "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Linear Regression"])

# Model training and prediction
if model_name == "K-Nearest Neighbors":
    model = KNeighborsClassifier(n_neighbors=3)
    explanation = "K-Nearest Neighbors (KNN) classifies data points based on the majority class of their nearest neighbors."
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
    explanation = "Decision Tree is a model that splits the data into branches to make predictions based on the values of input features."
elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    explanation = "Random Forest is an ensemble method that builds multiple decision trees and merges them together to get a more accurate and stable prediction."
else:
    model = LinearRegression()
    explanation = "Linear Regression predicts the target variable as a linear combination of the input features. Although typically used for continuous targets, here we use it for classification."

st.write("### Model Explanation")
st.write(explanation)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# If using Linear Regression, round the predictions and convert to integer type
if model_name == "Linear Regression":
    y_pred = y_pred.round().astype(int)

# Model evaluation
if model_name == "Linear Regression":
    st.write("### Model Evaluation")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
else:
    accuracy = accuracy_score(y_test, y_pred)
    st.write("### Model Evaluation")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")


# Data visualization based on predictions
if model_name == "Linear Regression":
    st.write("### Data Visualization (Linear Regression)")
    st.write("Linear Regression does not produce categorical predictions, hence no categorical scatter plots are available.")

    # Scatter plot of predicted vs true values
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue')
    ax.plot([y_test.min(), y_test.max()], [
            y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('True vs Predicted Values (Linear Regression)')
    st.pyplot(fig)

else:
    st.write(f"### Data Visualization ({model_name})")
    # Add predictions to the dataset for visualization
    data['predicted_species'] = model.predict(X)

    # Scatter plot based on sepal measurements
    fig_sepal, ax_sepal = plt.subplots()
    sns.scatterplot(data=data, x='sepal_length', y='sepal_width',
                    hue='predicted_species', palette='viridis', ax=ax_sepal)
    ax_sepal.set_title('Sepal Measurements')
    st.pyplot(fig_sepal)

    # Scatter plot based on petal measurements
    fig_petal, ax_petal = plt.subplots()
    sns.scatterplot(data=data, x='petal_length', y='petal_width',
                    hue='predicted_species', palette='viridis', ax=ax_petal)
    ax_petal.set_title('Petal Measurements')
    st.pyplot(fig_petal)

# User input for predictions
st.write("### Predict Iris Species")
sepal_length = st.slider("Sepal Length", float(
    X['sepal_length'].min()), float(X['sepal_length'].max()))
sepal_width = st.slider("Sepal Width", float(
    X['sepal_width'].min()), float(X['sepal_width'].max()))
petal_length = st.slider("Petal Length", float(
    X['petal_length'].min()), float(X['petal_length'].max()))
petal_width = st.slider("Petal Width", float(
    X['petal_width'].min()), float(X['petal_width'].max()))

# Predict button
if st.button("Predict"):
    prediction = model.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    if model_name == "Linear Regression":
        prediction = prediction.round().astype(int)
    predicted_species = le.inverse_transform(prediction)[0]
    st.write(f"The predicted species is: {predicted_species}")