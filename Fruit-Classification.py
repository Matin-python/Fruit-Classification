import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_table('fruit.txt')
data = data.drop(['fruit_name', 'fruit_subtype'], axis=1)

feature_names = ['mass', 'width', 'height', 'color_score']
X = data[feature_names]
y = data['fruit_label']

# Visualize (SAVE before SHOW)
cmap = plt.get_cmap('gnuplot')
scatter = scatter_matrix(X, 
                         c = y,
                         s = 40,
                         figsize=(7,7),
                         hist_kwds={'bins':15},
                         marker= 'o',
                         cmap= cmap)
plt.suptitle('scatter-matrix for each input variable')

plt.savefig('screenshots/fruit_scatter_matrix.png')
plt.show()

# Split data (with random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size= 0.15, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression
log_model = LogisticRegression(max_iter= 1000000)
log_model.fit(X_train, y_train)

print('='*60)
print('Accuracy of Logistic Regression Classifier on training set: {:0.2f}' .format(log_model.score(X_train, y_train)))
print('Accuracy of Logistic Regression Classifier on test set: {:0.2f}' .format(log_model.score(X_test, y_test)))
print('='*60)


# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
print('Accuracy of Decision Tree Classifier on training set: {:0.2f}' .format(dt_model.score(X_train, y_train)))
print('Accuracy of Decision Tree Classifier on test set: {:0.2f}' .format(dt_model.score(X_test, y_test)))
print('='*60)


# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
print('Accuracy of KNeighbors Classifier on training set: {:0.2f}' .format(knn_model.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:0.2f}' .format(knn_model.score(X_test, y_test)))
print('='*60)


# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
print('Accuracy of Support vector machines(SVM) Classifier on training set: {:0.2f}' .format(svm_model.score(X_train, y_train)))
print('Accuracy of Support vector machines(SVM) Classifier on test set: {:0.2f}' .format(svm_model.score(X_test, y_test)))
