# Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random

# Set random seed for reproducibility
random.seed(100)

# Function to load and preprocess data
def load_and_preprocess_data(path):
    wine = pd.read_csv(path)
    
    # Binarize the 'quality' column into 'bad' and 'good'
    bins = (2, 6.5, 8)
    labels = ['bad', 'good']
    wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)

    # Encode 'bad' and 'good' to 0 and 1
    encoder = LabelEncoder()
    wine['quality'] = encoder.fit_transform(wine['quality'])

    X = wine.drop('quality', axis=1)
    y = wine['quality']
    return X, y, wine.columns[:11]

# Function to plot feature importances using Random Forest
def plot_feature_importance(X, y, feature_labels):
    model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature importances:")
    for i in range(X.shape[1]):
        print(f"{i+1:2d}) {feature_labels[i]:30} {importances[indices[i]]:.6f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_labels[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Function to scale and apply PCA
def scale_and_reduce(X_train, X_test, n_components=4):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_var = pd.DataFrame(pca.explained_variance_ratio_, columns=["Explained Variance Ratio"])
    print("PCA Explained Variance Ratio:")
    print(explained_var)

    return X_train_pca, X_test_pca

# Function to train and evaluate a model
def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, results_df):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_result = pd.DataFrame([[model_name, acc, prec, rec, f1]],
                                columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    return results_df.append(model_result, ignore_index=True)

# Main Execution Function
def main():
    # Load and preprocess data
    X, y, feature_labels = load_and_preprocess_data('data/winequality.csv')

    # Feature importance plot
    plot_feature_importance(X, y, feature_labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

    # Scale features and apply PCA
    X_train_pca, X_test_pca = scale_and_reduce(X_train, X_test)

    # Initialize results dataframe
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Evaluate different models
    results = evaluate_model("Logistic Regression", LogisticRegression(random_state=0),
                             X_train_pca, y_train, X_test_pca, y_test, results)

    results = evaluate_model("SVM (Linear)", SVC(kernel='linear', random_state=0),
                             X_train_pca, y_train, X_test_pca, y_test, results)

    results = evaluate_model("SVM (RBF)", SVC(kernel='rbf', random_state=0),
                             X_train_pca, y_train, X_test_pca, y_test, results)

    results = evaluate_model("Random Forest (n=100)", RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
                             X_train_pca, y_train, X_test_pca, y_test, results)

    # Display results
    print("\nModel Performance Comparison:")
    print(results)

# Run the script
if __name__ == "__main__":
    main()
