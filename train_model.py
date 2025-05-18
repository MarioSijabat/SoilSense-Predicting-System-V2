import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib

# Tentukan folder penyimpanan plot
folder_plots = "hasil_train_plots"
if not os.path.exists(folder_plots):
    os.makedirs(folder_plots)

# Load data
df = pd.read_csv("data_core.csv")
print(df.head())
print(df.info())
print(df.describe())

# Fitur numerik
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Histogram distribusi
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribusi {feature}')
plt.tight_layout()
plt.savefig(os.path.join(folder_plots, "plot_histogram_distribusi.png"))
plt.close()

# Heatmap korelasi
corr = df[features].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi Fitur Numerik')
plt.savefig(os.path.join(folder_plots, "plot_korelasi_heatmap.png"))
plt.close()

# Countplot label
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='label')
plt.title('Distribusi Kelas Label')
plt.savefig(os.path.join(folder_plots, "plot_distribusi_label.png"))
plt.close()

# Boxplot per fitur terhadap label
for feature in features:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df, x='label', y=feature)
    plt.title(f'{feature} berdasarkan Label')
    plt.savefig(os.path.join(folder_plots, f"boxplot_{feature}_vs_label.png"))
    plt.close()

# Pairplot
sns.pairplot(df[features + ['label']], hue='label')
plt.savefig(os.path.join(folder_plots, "plot_pairplot.png"))
plt.close()

# Skew dan Kurtosis
numeric_df = df.select_dtypes(include=['number'])
print("Skewness:\n", numeric_df.skew())
print("Kurtosis:\n", numeric_df.kurt())

# Missing value check
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values[missing_values > 0])

# Boxplot per fitur (independen dari label)
plt.figure(figsize=(15,10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot {feature}')
plt.tight_layout()
plt.savefig(os.path.join(folder_plots, "plot_boxplot_all_features.png"))
plt.close()

# Persiapan data
X = df[features]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Label classes:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Original dataset shape: {X_train_scaled.shape}, {y_train.shape}")
print(f"Resampled dataset shape: {X_res.shape}, {y_res.shape}")

# GridSearchCV untuk Decision Tree
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_res, y_res)

print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2%}")

# Evaluasi klasifikasi
model_classifier = grid_search.best_estimator_
y_pred = model_classifier.predict(X_test_scaled)

print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("[INFO] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(folder_plots, "plot_confusion_matrix.png"))
plt.close()

acc = model_classifier.score(X_test_scaled, y_test)
print(f"[INFO] Akurasi model klasifikasi: {acc:.2%}")

# Regressi untuk fertility_score
df['fertility_score'] = df['N'] * 0.4 + df['P'] * 0.3 + df['K'] * 0.3
y_regression = df["fertility_score"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42)

X_train_scaled_reg = scaler.fit_transform(X_train_reg)
X_test_scaled_reg = scaler.transform(X_test_reg)

model_regressor = LinearRegression()
model_regressor.fit(X_train_scaled_reg, y_train_reg)

y_pred_reg = model_regressor.predict(X_test_scaled_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"[INFO] Mean Squared Error regresi: {mse:.4f}")
print(f"[INFO] R2 score regresi: {r2:.2%}")

# Simpan model
joblib.dump(model_classifier, "model_classifier.pkl")
joblib.dump(model_regressor, "model_regressor.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("[INFO] Training selesai. Semua model dan plot berhasil disimpan.")