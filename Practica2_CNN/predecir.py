from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import load_img, img_to_array 
import numpy as np 
# Cargar el modelo entrenado
model = load_model('intento1_deportes.h5')

# Cargar y preprocesar la imagen para que coincida con el tamaño esperado por el modelo 
test_image = load_img('single_test/tenis.jpg', target_size=(64, 64))  # Cambia la ruta a tu imagen y ajusta el tamaño al que entrenaste (64x64)
test_image = img_to_array(test_image)

# Normalizar la imagen como en el entrenamiento 
test_image = test_image / 255.0

# Expandir las dimensiones para hacer la predicción (1, 64, 64, 3)
test_image = np.expand_dims(test_image, axis=0)

# Hacer la predicción
result = model.predict(test_image)

# Mostrar los valores predichos (probabilidades para cada clase)
print("Probabilidades predichas:", result)

# Obtener el índice de la clase con mayor probabilidad
predicted_class_index = np.argmax(result)

# Obtener el mapeo de clases (esto necesita el generador de datos de entrenamiento)
# Puedes guardar el mapeo de clases al momento de entrenamiento o cargarlo desde training_dataset.class_indices si está disponible
class_labels = {0: 'ajedrez', 1: 'baloncesto', 2: 'boxeo', 3: 'disparo', 4: 'esgrima',
                5: 'formula1', 6: 'futbol', 7: 'hockey', 8: 'natacion', 9: 'tenis'}  # Ejemplo de cómo podrías mapear las clases

# Obtener el nombre de la clase predicha
predicted_class_label = class_labels[predicted_class_index]

# Imprimir la clase predicha
print(f'La imagen pertenece a la clase: {predicted_class_label}')
