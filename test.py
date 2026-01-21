import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

MODEL_PATH = '../tmp/current_model/model.h5'
IMG_SIZE = 224
CLASS_NAMES = [
    "BacterialBlights",
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow"
  ] 

def square_resize_image(img_input, target_size):
    # TRAINING-LIKE STANDARIZATION
    img_out = img_input
    h, w = img_out.shape[:2]
    
    # Padding to square-shaping the image
    top, bottom, left, right = 0, 0, 0, 0
    if w >= h:
        top = (w - h) // 2
        bottom = (w - h) - top
    else:
        left = (h - w) // 2
        right = (h - w) - left
        
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        img_out = cv.copyMakeBorder(img_out, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    
    # 2. Resize
    img_out = cv.resize(img_out, (target_size, target_size), interpolation=cv.INTER_AREA)
    return img_out

def load_and_preprocess_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: No se pudo leer la imagen {image_path}")
        return None

    # 1. Sizing
    img = square_resize_image(img, IMG_SIZE)

    # 2. BGR -> RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype('float32')

    # 4. Adding an extra dimension (1, 224, 224, 3)
    img_expanded = np.expand_dims(img, axis=0)
    
    return img_expanded

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encuentra el modelo en {MODEL_PATH}")
        return

    print(f"Cargando modelo desde {MODEL_PATH}...")
    
    try:
        model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})
    except Exception as e:
        print(f"\nERROR CRÍTICO AL CARGAR:\n{e}")
        return

    print("Modelo cargado exitosamente.")

    while True:
        image_path = input("\nIntroduce la ruta de la imagen (o 'q' para salir): ").strip()
        image_path = image_path.replace('"', '').replace("'", "")

        if image_path.lower() == 'q':
            break

        if not os.path.exists(image_path):
            print("La ruta no existe.")
            continue

        processed_img = load_and_preprocess_image(image_path)
        
        if processed_img is not None:
            # Predicting
            predictions = model.predict(processed_img, verbose=0)
            
            # Translating
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            if predicted_class_idx < len(CLASS_NAMES):
                class_name = CLASS_NAMES[predicted_class_idx]
            else:
                class_name = f"Clase {predicted_class_idx} (Desconocida)"

            print(f"------------------------------------------------")
            print(f"PREDICCIÓN: {class_name}")
            print(f"CONFIANZA:  {confidence:.2f}%")
            print(f"------------------------------------------------")

if __name__ == "__main__":
    main()