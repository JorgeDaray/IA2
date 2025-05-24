import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Flatten
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Cargar los datos del dataset
train_data = pd.read_csv(r'mitbih_train.csv', header=None)
test_data = pd.read_csv(r'mitbih_test.csv', header=None)

# Verificar el tamaño de los datos
print(f"train data size: {train_data.shape}")
print(f"test data size: {test_data.shape}")

# Preparar los datos
data = train_data.iloc[:, :187]
labels = train_data.iloc[:, 187]

# Balancear los datos
ovrs = RandomOverSampler(random_state=42)
data_resampled, labels_resampled = ovrs.fit_resample(data, labels)
train_df = pd.concat([data_resampled, labels_resampled], axis=1)

# Dividir los datos en entrenamiento y validación
x = train_df.iloc[:, :187]
y = train_df.iloc[:, 187]

x_test = test_data.iloc[:, :187]
y_test = test_data.iloc[:, 187]

# Dividir el dataset en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=train_df.iloc[:, 187])

# Darle forma a los datos para el modelo CNN
x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)  # Convertir en una matriz 3D
x_val = x_val.values.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)

# Convertir las etiquetas a formato one-hot
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Crear el modelo CNN
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(x_train.shape[1], 1)),  # Capa convolucional con 32 filtros
    BatchNormalization(),  # Normalización por lotes
    MaxPooling1D(2),  # MaxPooling para reducir la dimensionalidad
    Dropout(0.3),  # Capa de Dropout para evitar sobreajuste

    Conv1D(64, 3, activation='relu'),  # Capa convolucional con 64 filtros
    BatchNormalization(),  # Normalización por lotes
    MaxPooling1D(2),  # MaxPooling
    Dropout(0.3),  # Dropout

    Conv1D(128, 3, activation='relu'),  # Capa convolucional con 128 filtros
    BatchNormalization(),  # Normalización por lotes
    MaxPooling1D(2),  # MaxPooling
    Dropout(0.3),  # Dropout

    Flatten(),  # Aplanar las salidas para la capa densa
    Dense(128, activation='relu'),  # Capa densa con 128 neuronas
    Dropout(0.5),  # Dropout

    Dense(5, activation='softmax')  # Capa de salida con 5 neuronas (una por cada clase)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definir callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('cnn_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Graficar la precisión y la pérdida durante el entrenamiento
plt.figure(figsize=(12, 5))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Epochs')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# Matriz de confusión
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[labels[i] for i in range(5)], yticklabels=[labels[i] for i in range(5)])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Reporte de clasificación
print(classification_report(y_true, y_pred_classes, target_names=labels.values()))
