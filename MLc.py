import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
df = pd.read_csv(r'K:\3rd year\MLcement.csv')

# Data Visualization
print("Columns in the dataset:", df.columns)

# Pairplot of Features
sns.pairplot(df, hue='CompressiveStrength')  # Adjust if needed
plt.title('Pairplot of Features')
plt.show()

# Distribution of Compressive Strength
sns.histplot(df['CompressiveStrength'], kde=True)
plt.title('Distribution of Compressive Strength')
plt.show()

# Feature and target separation
X = df.drop('CompressiveStrength', axis=1)
y = df['CompressiveStrength']

# Convert y to categorical if it's continuous
if y.dtype in [np.float64, np.int64]:
    y = pd.cut(y, bins=5, labels=False)

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Feature Selection with SelectKBest
k_best = SelectKBest(score_func=f_classif, k=5)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)
selected_features_kbest = np.where(k_best.get_support())[0]
print("SelectKBest selected features:", selected_features_kbest)

# Feature Selection with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
X_train_rf = X_train[:, indices[:5]]
X_test_rf = X_test[:, indices[:5]]
print("Random Forest selected features indices:", indices[:5])

# Initialize dictionaries to store results
results = {
    'KNN (Original)': {},
    'SVM (Original)': {},
    'KNN (SelectKBest)': {},
    'SVM (SelectKBest)': {},
    'KNN (RF)': {},
    'SVM (RF)': {}
}

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
results['KNN (Original)']['Accuracy'] = accuracy_score(y_test, y_pred_knn)
results['KNN (Original)']['Confusion Matrix'] = confusion_matrix(y_test, y_pred_knn)
results['KNN (Original)']['Classification Report'] = classification_report(y_test, y_pred_knn)

# SVM Classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
results['SVM (Original)']['Accuracy'] = accuracy_score(y_test, y_pred_svm)
results['SVM (Original)']['Confusion Matrix'] = confusion_matrix(y_test, y_pred_svm)
results['SVM (Original)']['Classification Report'] = classification_report(y_test, y_pred_svm)

# KNN with SelectKBest
knn.fit(X_train_kbest, y_train)
y_pred_knn_kbest = knn.predict(X_test_kbest)
results['KNN (SelectKBest)']['Accuracy'] = accuracy_score(y_test, y_pred_knn_kbest)
results['KNN (SelectKBest)']['Confusion Matrix'] = confusion_matrix(y_test, y_pred_knn_kbest)
results['KNN (SelectKBest)']['Classification Report'] = classification_report(y_test, y_pred_knn_kbest)

# SVM with SelectKBest
svm.fit(X_train_kbest, y_train)
y_pred_svm_kbest = svm.predict(X_test_kbest)
results['SVM (SelectKBest)']['Accuracy'] = accuracy_score(y_test, y_pred_svm_kbest)
results['SVM (SelectKBest)']['Confusion Matrix'] = confusion_matrix(y_test, y_pred_svm_kbest)
results['SVM (SelectKBest)']['Classification Report'] = classification_report(y_test, y_pred_svm_kbest)

# KNN with Random Forest feature selection
knn.fit(X_train_rf, y_train)
y_pred_knn_rf = knn.predict(X_test_rf)
results['KNN (RF)']['Accuracy'] = accuracy_score(y_test, y_pred_knn_rf)
results['KNN (RF)']['Confusion Matrix'] = confusion_matrix(y_test, y_pred_knn_rf)
results['KNN (RF)']['Classification Report'] = classification_report(y_test, y_pred_knn_rf)

# SVM with Random Forest feature selection
svm.fit(X_train_rf, y_train)
y_pred_svm_rf = svm.predict(X_test_rf)
results['SVM (RF)']['Accuracy'] = accuracy_score(y_test, y_pred_svm_rf)
results['SVM (RF)']['Confusion Matrix'] = confusion_matrix(y_test, y_pred_svm_rf)
results['SVM (RF)']['Classification Report'] = classification_report(y_test, y_pred_svm_rf)

# Data Visualization

# 1. Feature Importance from Random Forest
plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Random Forest")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# 2. Accuracy Plot
plt.figure(figsize=(12, 6))
model_labels = list(results.keys())
accuracy_values = [results[model]['Accuracy'] for model in results]

sns.barplot(x=accuracy_values, y=model_labels, palette='viridis')
plt.xlabel('Accuracy')
plt.title('Accuracy for Different Models and Feature Selection Techniques')
plt.show()

# 3. Confusion Matrix Plots
def plot_confusion_matrix(cm, ax, title='Confusion Matrix', cmap=plt.cm.Blues):
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for i, (key, value) in enumerate(results.items()):
    row, col = divmod(i, 3)
    cm = value['Confusion Matrix']
    plot_confusion_matrix(cm, ax=ax[row, col], title=key)
plt.tight_layout()
plt.show()

# Print Classification Reports
for model in results:
    print(f"\nClassification Report for {model}:\n{results[model]['Classification Report']}")
