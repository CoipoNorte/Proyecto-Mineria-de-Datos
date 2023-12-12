import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Cambio de modelo
from sklearn.metrics import accuracy_score, classification_report

# Modelo 2: Carga de Datos
data_model2 = pd.read_csv('fatalities_isr_pse_conflict_2000_to_2023.csv')  # 2.1

# Análisis Exploratorio de Datos (EDA) para Modelo 2
# (Se utiliza la biblioteca Pandas y Seaborn para visualizar y entender el conjunto de datos)

# Visualizar las primeras filas del conjunto de datos
print("Primeras filas del conjunto de datos (Modelo 2):")
print(data_model2.head())

# Información general sobre el conjunto de datos
print("\nInformación general sobre el conjunto de datos (Modelo 2):")
print(data_model2.info())

# Estadísticas descriptivas del conjunto de datos
print("\nEstadísticas descriptivas (Modelo 2):")
print(data_model2.describe())

# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(8, 6))
sns.countplot(x='killed_by', data=data_model2)
plt.title('Distribución de la variable objetivo (Modelo 2)')
plt.show()

# Visualización de la relación entre algunas características
plt.figure(figsize=(12, 8))
sns.pairplot(data_model2[['age', 'took_part_in_the_hostilities', 'type_of_injury', 'killed_by']], hue='killed_by')
plt.title('Relación entre algunas características (Modelo 2)')
plt.show()

# Matriz de correlación
correlation_matrix_model2 = data_model2.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_model2, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación (Modelo 2)')
plt.show()

# Modelo 2: Creación del Nuevo Modelo (K Vecinos Más Cercanos - KNN)
X_model2 = data_model2.drop('killed_by', axis=1)  # 2.2
y_model2 = data_model2['killed_by']

X_train_model2, X_test_model2, y_train_model2, y_test_model2 = train_test_split(X_model2, y_model2, test_size=0.2, random_state=42)  # 2.2

scaler_model2 = StandardScaler()  # 2.2
X_train_model2 = scaler_model2.fit_transform(X_train_model2)
X_test_model2 = scaler_model2.transform(X_test_model2)

model2 = KNeighborsClassifier(n_neighbors=5)  # Cambio de modelo a KNN (2.2)
model2.fit(X_train_model2, y_train_model2)

predictions_model2 = model2.predict(X_test_model2)
accuracy_model2 = accuracy_score(y_test_model2, predictions_model2)

# Análisis y Predicción (Modelo 2)
print("Modelo 2 (Nuevo Modelo) - Accuracy:", accuracy_model2)
print("Classification Report Modelo 2 (Nuevo Modelo):")
print(classification_report(y_test_model2, predictions_model2))
