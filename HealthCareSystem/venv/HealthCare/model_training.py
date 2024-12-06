import json
import time
import pandas as pd
from preprocess import preprocess_text
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pickle
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve
import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import learning_curve
import warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore")


# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Function to plot confusion matrix


def plot_confusion_matrix(y_true, y_pred, model_name, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'static/models/ConfusionMatrix/{model_name} Confusion Matrix')
    plt.show()

# Function to plot learning curves


def plot_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1),
             label='Cross-Validation Score')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve - {model_name}")
    plt.legend()
    plt.savefig(f'static/models/LearningCurves/{model_name} Learning Curve')
    plt.show()


def plot_calibration_curve(model, X_test, y_test, classes, model_name):
    plt.figure(figsize=(12, 8))

    # Binarize the labels
    y_test_binarized = label_binarize(y_test, classes=np.arange(len(classes)))

    for i, class_name in enumerate(classes):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                X_test)[:, i]  # Probabilities for class i
        else:
            print(f"Model {model_name} does not support predict_proba.")
            return

        prob_true, prob_pred = calibration_curve(
            y_test_binarized[:, i], probabilities, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f"Class {class_name}")
    # Add diagonal line and labels
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="best")
    plt.savefig(
        f"static/models/CalibrationCurves/{model_name}_calibration_curve.png")
    plt.show()


# Set up NLTK data path
nltk.data.path.append(
    '/Users/udaydeepreddy/Desktop/HealthCareSystem/venv/HealthCare/nltk_data/tokenizers/')

# Load intents data
with open("intents.json") as file:
    intents = json.load(file)

# Prepare data for training
data = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        data.append({'tag': tag, 'pattern': preprocess_text(pattern)})
df = pd.DataFrame(data)


# Vectorize patterns and encode labels
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['pattern']).toarray()
encoder = LabelEncoder()
y = encoder.fit_transform(df['tag'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

bayes_model = MultinomialNB()

# Generate alpha values from 0 to 1 with three-digit precision
alphas = np.linspace(0.001, 1.000, 1000)  # 1000 values between 0.001 and 1.000

# Define the parameter grid
param_grid = {'alpha': alphas}

# Define GridSearchCV
grid_search = GridSearchCV(
    estimator=bayes_model,
    param_grid=param_grid,
    scoring=make_scorer(accuracy_score),  # Use accuracy as the scoring metric
    cv=5,  # 5-fold cross-validation
    verbose=1,  # Show progress
    n_jobs=-1  # Use all available cores
)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Store the best alpha value and its corresponding score
best_alpha = grid_search.best_params_['alpha']
best_score = grid_search.best_score_

print(f"Best alpha value: {best_alpha:.3f}")
print(f"Best cross-validation accuracy: {best_score:.4f}")

# Train and evaluate Naive Bayes
naive_bayes_model = MultinomialNB(alpha=best_alpha)

naive_bayes_model.fit(X_train, y_train)
y_train_pred_nb = naive_bayes_model.predict(X_train)
y_test_pred_nb = naive_bayes_model.predict(X_test)
nb_train_accuracy = accuracy_score(y_train, y_train_pred_nb)
nb_test_accuracy = accuracy_score(y_test, y_test_pred_nb)


# Calibration curves for Naive Bayes
plot_calibration_curve(naive_bayes_model, X_test, y_test,
                       encoder.classes_, "Naive Bayes")

# Plot for Naive Bayes
plot_confusion_matrix(y_test, y_test_pred_nb, "Naive Bayes", encoder.classes_)

plot_learning_curve(naive_bayes_model, X, y, "Naive Bayes")
precision_score_nb = precision_score(
    y_test, y_test_pred_nb, average='weighted')
recall_score_nb = recall_score(y_test, y_test_pred_nb, average='weighted')
nb_f1_score = f1_score(y_test, y_test_pred_nb, average='weighted')
print("\nNaive Bayes Results:")
print(f"Training Accuracy: {nb_train_accuracy}")
print(f"Test Accuracy: {nb_test_accuracy}")
print("\nTest Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_nb))
print("\nTest Classification Report:\n", classification_report(
    y_test, y_test_pred_nb, target_names=encoder.classes_, labels=encoder.transform(
        encoder.classes_)
))
print(f" Precision Score: {precision_score_nb}")
print(f" Recall Score: {recall_score_nb}")
print(f" F1 Score: {nb_f1_score}")

# Train and evaluate Logistic Regression
logi_model = LogisticRegression()

# Define the hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 4, 10),  # Values for regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization types
    'solver': ['saga'],  # Choose solvers compatible with penalties
    'max_iter': [100, 200, 300]  # Vary the max number of iterations
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=logi_model, param_grid=param_grid, 
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Best Parameters: {best_params}")
print(f"Test Accuracy with Best Model: {test_accuracy}")
##  Best Parameters for Logistic Regression
logistic_model = LogisticRegression(
    penalty=best_params['penalty'],
    C=best_params['C'],
    solver=best_params['solver'],
    max_iter=best_params['max_iter']
)

logistic_model.fit(X_train, y_train)
y_train_pred_lr = logistic_model.predict(X_train)
y_test_pred_lr = logistic_model.predict(X_test)
lr_train_accuracy = accuracy_score(y_train, y_train_pred_lr)
lr_test_accuracy = accuracy_score(y_test, y_test_pred_lr)

# Calibration curves for Logistic Regression
plot_calibration_curve(logistic_model, X_test, y_test,
                       encoder.classes_, "Logistic Regression")

# Plot for Logistic Regression
plot_confusion_matrix(y_test, y_test_pred_lr,
                      "Logistic Regression", encoder.classes_)

plot_learning_curve(logistic_model, X, y, "Logistic Regression")

precision_score_lr = precision_score(
    y_test, y_test_pred_lr, average='weighted')
recall_score_lr = recall_score(y_test, y_test_pred_lr, average='weighted')
lr_f1_score = f1_score(y_test, y_test_pred_lr, average='weighted')
print("\nLogistic Regression Results:")
print(f"Training Accuracy: {lr_train_accuracy}")
print(f"Test Accuracy: {lr_test_accuracy}")
print("\nTest Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_lr))
print("\nTest Classification Report:\n", classification_report(
    y_test, y_test_pred_lr, target_names=encoder.classes_, labels=encoder.transform(
        encoder.classes_)
))
print(f" Precision Score: {precision_score_lr}")
print(f" Recall Score: {recall_score_lr}")
print(f" F1 Score: {lr_f1_score}")

# Train and evaluate Neural Network
# Generate alpha values from 0.001 to 1.0 with increments of 0.001


# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': (300, 200, 150, 100, 50, 25),
    'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
    'solver': ['adam', 'sgd'],                                     # Optimization solvers
    'alpha': [0.0001, 0.001, 0.01, 0.1],                                                  # Regularization strengths
    'learning_rate': ['constant', 'adaptive'],                # Learning rate strategies
    'max_iter': [200, 300, 500, 1000, 2000]                                 # Maximum iterations
}
"""
# Initialize the MLPClassifier
mlp = MLPClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Use the best estimator
best_mlp_model = grid_search.best_estimator_
"""
best_para= {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (300,150,100, 50), 
            'learning_rate': 'constant', 'max_iter': 2000, 'solver': 'sgd'}
print(best_para)
neural_net_model = MLPClassifier(
    hidden_layer_sizes=best_para['hidden_layer_sizes'],
    activation=best_para['activation'],
    solver=best_para['solver'],
    alpha=best_para['alpha'],
    learning_rate=best_para['learning_rate'],
    max_iter=best_para['max_iter'],
    random_state=42
)
neural_net_model.fit(X_train, y_train)
y_train_pred_nn = neural_net_model.predict(X_train)
y_test_pred_nn = neural_net_model.predict(X_test)
nn_train_accuracy = accuracy_score(y_train, y_train_pred_nn)
nn_test_accuracy = accuracy_score(y_test, y_test_pred_nn)

# Plot for Neural Network
plot_confusion_matrix(y_test, y_test_pred_nn,
                      "Neural Network", encoder.classes_)

plot_learning_curve(neural_net_model, X, y, "Neural Network")
# Calibration curves for Neural Network
plot_calibration_curve(neural_net_model, X_test, y_test,
                       encoder.classes_, "Neural Network")

precision_score_nn = precision_score(
    y_test, y_test_pred_nn, average='weighted')
recall_score_nn = recall_score(y_test, y_test_pred_nn, average='weighted')
nn_f1_score = f1_score(y_test, y_test_pred_nn, average='weighted')


print("\nNeural Network Results:")
print(f"Training Accuracy: {nn_train_accuracy}")
print(f"Test Accuracy: {nn_test_accuracy}")
print("\nTest Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_nn))
print("\nTest Classification Report:\n", classification_report(
    y_test, y_test_pred_nn, target_names=encoder.classes_, labels=encoder.transform(
        encoder.classes_)
))
print(f" Precision Score: {precision_score_nn}")
print(f" Recall Score: {recall_score_nn}")
print(f" F1 Score: {nn_f1_score}")

# Determine the best model based on test accuracy
best_model = None
best_model_name = ""
best_test_accuracy = max(nb_test_accuracy, lr_test_accuracy, nn_test_accuracy)

if best_test_accuracy == nb_test_accuracy:
    best_model = naive_bayes_model
    best_model_name = "Naive Bayes"
elif best_test_accuracy == lr_test_accuracy:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
else:
    best_model = neural_net_model
    best_model_name = "Neural Network"

print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {best_test_accuracy}")

# Save the best model, vectorizer, and encoder
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Graphs for comparison of Train and Test Accuracy of these two modals
# Replace these with actual calculated accuracies from your models
model_names = ["Naive Bayes", "Logistic Regression", "Neural Network"]
# Replace with actual training accuracies
train_accuracies = [nb_train_accuracy, lr_train_accuracy, nn_train_accuracy]
test_accuracies = [nb_test_accuracy, lr_test_accuracy,
                   nn_test_accuracy]   # Replace with actual test accuracies

# Plot train and test accuracies
plt.figure(figsize=(10, 6))
x = np.arange(len(model_names))  # Label locations
width = 0.35  # Width of the bars

# Bar plots
plt.bar(x - width/2, train_accuracies, width,
        label='Train Accuracy', alpha=0.7)
plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', alpha=0.7)

# Add labels and titles
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Comparison of Train and Test Accuracies Across Models")
plt.xticks(x, model_names)
plt.ylim(0, 1)
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig("static/models/Accuracy_Models.png")
plt.show()

precision_scores = [
    precision_score_nb,
    precision_score_lr,
    precision_score_nn,
]

recall_scores = [
    recall_score_nb,
    recall_score_lr,
    recall_score_nn,
]

f1_scores = [
    nb_f1_score,
    lr_f1_score,
    nn_f1_score,
]

# Corrected Bar Chart Code
plt.figure(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(model_names))  # Correctly set x positions for the number of models

# Create bars for Precision, Recall, and F1-Score
plt.bar(x - bar_width, precision_scores, width=bar_width, label='Precision', alpha=0.8)
plt.bar(x, recall_scores, width=bar_width, label='Recall', alpha=0.8)
plt.bar(x + bar_width, f1_scores, width=bar_width, label='F1-Score', alpha=0.8)

# Add labels and titles
plt.xlabel("Models", fontweight='bold')
plt.ylabel("Scores", fontweight='bold')
plt.title("Precision, Recall, and F1-Score Comparison Across Models", fontweight='bold')
plt.xticks(x, model_names)  # Use the correct labels for x positions
plt.ylim(0, 1)
plt.legend()

# Save and display the plot
plt.tight_layout()
plt.savefig("static/models/precision_recall_f1_comparison.png")
plt.show()
