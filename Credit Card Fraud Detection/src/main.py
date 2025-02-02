import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri Setini Yükleme
data_path = './data/creditcard.csv'  # Veri setinin yolu
df = pd.read_csv(data_path)

# Veri hakkında bilgi
print("Veri Seti Bilgileri:")
print(df.info())

# İlk 5 satırı görüntüleme
print("İlk 5 Satır:")
print(df.head())

# 2. Hedef ve Özellikleri Ayırma
X = df.drop('Class', axis=1)  # Özellikler
y = df['Class']              # Hedef değişken

# 3. Eğitim ve Test Verisi Olarak Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Özellikleri Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. KNN Modeli ile Eğitim
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 6. Test Verisi ile Tahmin
y_pred = knn.predict(X_test)

# 7. Modelin Performansını Değerlendirme
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix Görselleştirme
ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.title('KNN Model - Confusion Matrix')
plt.show()

# 9. Model Performansını Görselleştirme (Opsiyonel)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('KNN Model - Confusion Matrix Heatmap')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()
