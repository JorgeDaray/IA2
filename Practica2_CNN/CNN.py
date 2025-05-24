import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Directorio de entrenamiento
train_dir = 'dataset'

# datos con aumento de datos para el entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Escalar los valores de los píxeles a [0, 1]
    shear_range=0.2,         # Aplicar transformaciones de corte
    zoom_range=0.2,          # Aplicar zoom a las imágenes
    horizontal_flip=True,     # Voltear horizontalmente
    validation_split=0.2)     # Separar el 20% de los datos para validación

# datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),    # Cambiar tamaño de imágenes a 64x64 píxeles ajustar dependiendo el CNN (explicado en pdf)
    batch_size=32,
    class_mode='categorical',
    subset='training')       # Usar los datos para entrenamiento

# datos de validación
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation')     # Usar los datos para validación

# Definir el modelo secuencial
model = Sequential()

# Capa convolucional 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Capa convolucional 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Capa convolucional 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Aplanar las capas convolucionales
model.add(Flatten())

# Capa densa 1
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(10, activation='softmax'))  # Cambiar por el número de clases deportivas (en este caso 10)

# Adam ajustado y categorical_crossentropy
model.compile(optimizer=Adam(learning_rate=0.0001),  # tasa de aprendizaje 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# Resumen del modelo
model.summary()

# Guardar el modelo entrenado (opcional)
model.save('intento1_deportes.h5')

# Evaluación del modelo con métricas adicionales

# Reiniciar el generador de validación
validation_generator.reset()

# Obtener etiquetas verdaderas y predicciones del modelo 
Y_true = validation_generator.classes
Y_pred = model.predict(validation_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Etiquetas de las clases
class_labels = list(validation_generator.class_indices.keys())

# Reporte de clasificación para el conjunto de validación
print("Reporte de clasificación para el conjunto de validación:\n")
print(classification_report(Y_true, Y_pred_classes, target_names=class_labels))

# Matriz de confusión
conf_matrix = confusion_matrix(Y_true, Y_pred_classes)
print("Matriz de Confusión:\n", conf_matrix)

# Evaluación de precisión y pérdida en entrenamiento y validación
train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"\nPérdida en entrenamiento: {train_loss:.4f}, Precisión en entrenamiento: {train_accuracy:.4f}")
print(f"Pérdida en validación: {val_loss:.4f}, Precisión en validación: {val_accuracy:.4f}")
