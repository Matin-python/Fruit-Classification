# 🍎 Fruit Classification using Machine Learning

A Machine Learning project that classifies different types of fruit using four classification algorithms: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).


## Overview

This project uses a fruit dataset containing numerical measurements of different fruits. The features include mass, width, height, and color score.

The dataset is processed and divided into training and testing sets. Feature scaling is performed using `MinMaxScaler`, and four different Machine Learning classification algorithms are trained and evaluated using both training and test accuracy.

The project also includes a scatter matrix visualization to explore the relationships between the input features.


## Features

* Fruit classification
* Data preparation and feature selection
* Scatter matrix visualization
* Feature scaling using Min-Max normalization
* Logistic Regression classification
* Decision Tree classification
* K-Nearest Neighbors classification
* Support Vector Machine classification
* Training and test accuracy evaluation


## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn


## Dataset

The project uses the following dataset:

```text
fruit.txt
```

The dataset contains information about different types of fruit.

The following features are used for classification:

```text
mass
width
height
color_score
```

The target variable is:

```text
fruit_label
```

which represents the fruit class.

The columns `fruit_name` and `fruit_subtype` are removed before training because they are not used as input features.


## Data Preprocessing

The following preprocessing steps are performed:

### Feature Selection

The project uses four numerical features:

```python
feature_names = ['mass', 'width', 'height', 'color_score']
X = data[feature_names]
y = data['fruit_label']
```

### Train/Test Split

The dataset is divided into training and testing sets using an 85/15 split.

```text
85% → Training data
15% → Testing data
```

A fixed `random_state=42` is used so that the same split can be reproduced.

### Feature Scaling

`MinMaxScaler` is used to scale the input features before training the models.

```python
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

This transforms the feature values to a common scale and is particularly useful for models such as KNN, Logistic Regression, and SVM.


## Data Visualization

A scatter matrix is created to visualize the relationships between the input features.

```python
scatter = scatter_matrix(
    X,
    c=y,
    s=40,
    figsize=(7,7),
    hist_kwds={'bins':15},
    marker='o',
    cmap=cmap
)

plt.suptitle('scatter-matrix for each input variable')
plt.savefig('fruit_scatter_matrix')
plt.show()
```

The scatter matrix displays pairwise relationships between the features together with the distribution of each individual feature.

### Scatter Matrix

<p align="center">
  <img src="screenshots/fruit_scatter_matrix.png" width="1000">
</p>


## Machine Learning Workflow

The project follows the workflow below:

```text
Fruit Dataset
      ↓
Remove Unused Columns
      ↓
Select Input Features
      ↓
Create Scatter Matrix
      ↓
Train/Test Split
      ↓
Min-Max Feature Scaling
      ↓
Train Classification Models
      ↓
Make Predictions
      ↓
Evaluate Training and Test Accuracy
```


## Models

Four different Machine Learning classification algorithms are used in this project.

### 1. Logistic Regression

Logistic Regression is used to classify the fruit based on the selected numerical features.

```python
log_model = LogisticRegression(max_iter=1000000)
log_model.fit(X_train, y_train)
```

The training and test accuracy are calculated using the model's `score()` method.


### 2. Decision Tree

A Decision Tree classifier is used as another approach for fruit classification.

```python
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
```

The model is evaluated on both the training and testing datasets.


### 3. K-Nearest Neighbors (KNN)

K-Nearest Neighbors is used to classify fruit based on the nearest training samples.

```python
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
```

The model is evaluated using training and test accuracy.


### 4. Support Vector Machine (SVM)

Support Vector Machine is used as the fourth classification algorithm.

```python
svm_model = SVC()
svm_model.fit(X_train, y_train)
```

The training and test accuracy are then calculated.


## Model Training

Each classifier is trained using the same training dataset and evaluated using the same test dataset.

For each model, the project prints:

* Training accuracy
* Test accuracy

For example:

```python
print(
    'Accuracy of Logistic Regression Classifier on training set: {:0.2f}'
    .format(log_model.score(X_train, y_train))
)

print(
    'Accuracy of Logistic Regression Classifier on test set: {:0.2f}'
    .format(log_model.score(X_test, y_test))
)
```

The same evaluation process is applied to the Decision Tree, KNN, and SVM models.


## Evaluation

The models are evaluated using **accuracy scores**.

Two accuracy values are reported for each classifier:

* **Training Accuracy** — measures performance on the training dataset.
* **Test Accuracy** — measures performance on unseen test data.

### Example Output

```text
============================================================
Accuracy of Logistic Regression Classifier on training set: 0.XX
Accuracy of Logistic Regression Classifier on test set: 0.XX
============================================================
Accuracy of Decision Tree Classifier on training set: 1.00
Accuracy of Decision Tree Classifier on test set: 0.XX
============================================================
Accuracy of KNeighbors Classifier on training set: 0.XX
Accuracy of KNeighbors Classifier on test set: 0.XX
============================================================
Accuracy of Support vector machines(SVM) Classifier on training set: 0.XX
Accuracy of Support vector machines(SVM) Classifier on test set: 0.XX
```

The exact results depend on the dataset and the train/test split.


## License

This project is licensed under the **MIT License**.


## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Computer Vision, Machine Learning, and Artificial Intelligence.
