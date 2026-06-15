# ==========================================
# PREDIKSI HUJAN MENGGUNAKAN RANDOM FOREST
# ==========================================

# ==========================================
# 1. IMPORT LIBRARY
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 2. LOAD DATA
# ==========================================
df = pd.read_csv("Curah_Hujan.csv")

print("5 Data Pertama:")
print(df.head())

print("\nInformasi Dataset:")
print(df.info())

print("\nStatistik Deskriptif:")
print(df.describe())

# ==========================================
# 3. EDA & VISUALISASI
# ==========================================

# Cek Missing Value
print("\nMissing Value:")
print(df.isnull().sum())

# Distribusi Target
plt.figure(figsize=(6,4))
sns.countplot(x='Rain', data=df)
plt.title('Distribusi Kelas Rain')
plt.show()

# Histogram Semua Fitur
df.hist(figsize=(12,8))
plt.suptitle("Distribusi Setiap Fitur")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# 4. PREPROCESSING
# ==========================================

# Encoding target
le = LabelEncoder()
df['Rain'] = le.fit_transform(df['Rain'])
# no rain = 0
# rain = 1

# Pisahkan fitur dan target
X = df.drop('Rain', axis=1)
y = df['Rain']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah Data Training :", len(X_train))
print("Jumlah Data Testing  :", len(X_test))

# ==========================================
# 5. TERAPKAN METODE (RANDOM FOREST)
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# ==========================================
# 6. EVALUASI MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAkurasi Model :", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# ==========================================
# 7. VISUALISASI HASIL
# ==========================================

# Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Rain', 'Rain'],
    yticklabels=['No Rain', 'Rain']
)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

# Feature Importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(8,5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)
plt.title('Feature Importance Random Forest')
plt.show()