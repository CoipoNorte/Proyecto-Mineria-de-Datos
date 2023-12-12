import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Modelo 1: Carga de Datos
data_model1 = pd.read_csv('fatalities_isr_pse_conflict_2000_to_2023.csv')  # 1.1

# Análisis Exploratorio de Datos (EDA) para Modelo 1
# (Se utiliza la biblioteca Pandas y Seaborn para visualizar y entender el conjunto de datos)

# Visualizar las primeras filas del conjunto de datos
print("Primeras filas del conjunto de datos (Modelo 1):")
print(data_model1.head())

# Información general sobre el conjunto de datos
print("\nInformación general sobre el conjunto de datos (Modelo 1):")
print(data_model1.info())

# Estadísticas descriptivas del conjunto de datos
print("\nEstadísticas descriptivas (Modelo 1):")
print(data_model1.describe())

# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(8, 6))
sns.countplot(x='killed_by', data=data_model1)
plt.title('Distribución de la variable objetivo (Modelo 1)')
plt.show()

# Visualización de la relación entre algunas características
plt.figure(figsize=(12, 8))
sns.pairplot(data_model1[['age', 'took_part_in_the_hostilities', 'type_of_injury', 'killed_by']], hue='killed_by')
plt.title('Relación entre algunas características (Modelo 1)')
plt.show()

# Matriz de correlación
correlation_matrix_model1 = data_model1.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_model1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación (Modelo 1)')
plt.show()

# Modelo 1: Creación del Modelo
# (Aprendizaje de Máquina Supervisado - Bosque Aleatorio)
X_model1 = data_model1.drop('killed_by', axis=1)  # 1.2
y_model1 = data_model1['killed_by']

X_train_model1, X_test_model1, y_train_model1, y_test_model1 = train_test_split(X_model1, y_model1, test_size=0.2, random_state=42)  # 1.2

scaler_model1 = StandardScaler()  # 1.2
X_train_model1 = scaler_model1.fit_transform(X_train_model1)
X_test_model1 = scaler_model1.transform(X_test_model1)

model1 = RandomForestClassifier(random_state=42)  # 1.2
model1.fit(X_train_model1, y_train_model1)

predictions_model1 = model1.predict(X_test_model1)
accuracy_model1 = accuracy_score(y_test_model1, predictions_model1)

# Análisis y Predicción (Modelo 1)
print("Modelo 1 - Accuracy:", accuracy_model1)
print("Classification Report Modelo 1:")
print(classification_report(y_test_model1, predictions_model1))
