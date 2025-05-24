import tensorflow as tf
from keras._tf_keras.keras.applications import VGG16  
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Directorio de entrenamiento
train_dir = 'dataset'

# aumento de datos para el entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,          
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,     
    validation_split=0.2)

# datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),    
    batch_size=32,
    class_mode='categorical',
    subset='training')

# datos de validación
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Cargar el modelo preentrenado VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# capas convolucionales
for layer in base_model.layers:
    layer.trainable = False

# Añadir nuevas capas densas al final del modelo
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=output)

# Adam y categorical_crossentropy
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento inicial
history_initial = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
)

# capas del modelo base para fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compilar nuevamente el modelo para fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Realizar el fine-tuning
history_fine_tuning = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)]
)

# Guardar el modelo entrenado
model.save('modelo2_fine_tuning_deportes.h5')

# Evaluación del modelo inicial con métricas adicionales
print("\nEvaluación del modelo inicial:")
validation_generator.reset()
Y_true = validation_generator.classes
Y_pred_initial = model.predict(validation_generator, verbose=1)
Y_pred_classes_initial = np.argmax(Y_pred_initial, axis=1)

# Reporte de clasificación para el modelo inicial
class_labels = list(validation_generator.class_indices.keys())
print("Reporte de clasificación para el modelo inicial:\n")
print(classification_report(Y_true, Y_pred_classes_initial, target_names=class_labels))

# Matriz de confusión para el modelo inicial
conf_matrix_initial = confusion_matrix(Y_true, Y_pred_classes_initial)
print("Matriz de Confusión para el modelo inicial:\n", conf_matrix_initial)

# Evaluación de precisión y pérdida en entrenamiento y validación del modelo inicial
train_loss_initial, train_accuracy_initial = model.evaluate(train_generator, verbose=0)
val_loss_initial, val_accuracy_initial = model.evaluate(validation_generator, verbose=0)
print(f"\nPérdida en entrenamiento (inicial): {train_loss_initial:.4f}, Precisión en entrenamiento: {train_accuracy_initial:.4f}")
print(f"Pérdida en validación (inicial): {val_loss_initial:.4f}, Precisión en validación: {val_accuracy_initial:.4f}")

# Evaluación del modelo con fine-tuning con métricas adicionales
print("\nEvaluación del modelo con fine-tuning:")
validation_generator.reset()
Y_pred_fine_tuning = model.predict(validation_generator, verbose=1)
Y_pred_classes_fine_tuning = np.argmax(Y_pred_fine_tuning, axis=1)

# Reporte de clasificación para el modelo con fine-tuning
print("Reporte de clasificación para el modelo con fine-tuning:\n")
print(classification_report(Y_true, Y_pred_classes_fine_tuning, target_names=class_labels))

# Matriz de confusión para el modelo con fine-tuning
conf_matrix_fine_tuning = confusion_matrix(Y_true, Y_pred_classes_fine_tuning)
print("Matriz de Confusión para el modelo con fine-tuning:\n", conf_matrix_fine_tuning)

# Evaluación de precisión y pérdida en entrenamiento y validación del modelo con fine-tuning
train_loss_fine, train_accuracy_fine = model.evaluate(train_generator, verbose=0)
val_loss_fine, val_accuracy_fine = model.evaluate(validation_generator, verbose=0)
print(f"\nPérdida en entrenamiento (fine-tuning): {train_loss_fine:.4f}, Precisión en entrenamiento: {train_accuracy_fine:.4f}")
print(f"Pérdida en validación (fine-tuning): {val_loss_fine:.4f}, Precisión en validación: {val_accuracy_fine:.4f}")
