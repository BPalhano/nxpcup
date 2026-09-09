import os
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


DATA_FOLDER = "./deep_learning/resized_img_road"
INPUT_SHAPE = (200, 66, 3)

# ===============================
# Conversão ângulo → classe
# ===============================
def to_class(angle, thr=5.0):
    if angle < -thr:
        return 0   # Left
    elif angle > thr:
        return 2   # Right
    else:
        return 1   # Center

CLASS_NAMES = ["Left", "Center", "Right"]

# ===============================
# Carregamento + MEDIÇÃO resizing
# ===============================
def load_validation_data(data_folder, input_shape):
    csv_path = os.path.join(data_folder, "labels_converted.csv")

    images, labels = [], []
    resize_times = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            img_name, angle = row[0], float(row[1])
            img_path = os.path.join(data_folder, img_name)

            if not os.path.exists(img_path):
                continue

            t0 = time.perf_counter()

            img = Image.open(img_path).convert("RGB")
            img = img.resize((input_shape[1], input_shape[0]))
            img = np.asarray(img, dtype=np.float32) / 255.0

            t1 = time.perf_counter()
            resize_times.append(t1 - t0)

            images.append(img)
            labels.append(angle)

    X = np.array(images)
    y = np.array(labels)

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    resize_times = np.array(resize_times)

    print("==== TEMPO DE RESIZING ====")
    print(f"Médio: {resize_times.mean()*1000:.2f} ms")
    print(f"Desvio padrão: {resize_times.std()*1000:.2f} ms")
    print("==========================")

    return X_val, y_val

def show_misclassified_images(
    model,
    X_val,
    y_val,
    class_names,
    max_images=20
):
    inference_times = []
    y_pred_angle = []

    for img in X_val:
        input_tensor = np.expand_dims(img, axis=0)

        t0 = time.perf_counter()
        pred = model.predict(input_tensor, verbose=0)[0, 0]
        t1 = time.perf_counter()

        inference_times.append(t1 - t0)
        y_pred_angle.append(pred)

    y_pred_angle = np.array(y_pred_angle)
    inference_times = np.array(inference_times)

    print("==== TEMPO DE INFERÊNCIA (IA) ====")
    print(f"Médio: {inference_times.mean()*1000:.2f} ms")
    print(f"Desvio padrão: {inference_times.std()*1000:.2f} ms")
    print("================================")

    y_true_cls = np.array([to_class(a) for a in y_val])
    y_pred_cls = np.array([to_class(a) for a in y_pred_angle])

    bad_idx = np.where(y_true_cls != y_pred_cls)[0]

    print(f"Total de erros: {len(bad_idx)} / {len(y_val)}")

def save_misclassified(
    model,
    X_val,
    y_val,
    class_names,
    output_dir="misclassified"
):
    os.makedirs(output_dir, exist_ok=True)

    y_pred_angle = model.predict(X_val, verbose=0).flatten()

    y_true_cls = np.array([to_class(a) for a in y_val])
    y_pred_cls = np.array([to_class(a) for a in y_pred_angle])

    bad_idx = np.where(y_true_cls != y_pred_cls)[0]

    for i in bad_idx:
        img = (X_val[i] * 255).astype(np.uint8)
        gt = class_names[y_true_cls[i]]
        pred = class_names[y_pred_cls[i]]

        filename = (
            f"{i}_GT-{gt}_PRED-{pred}_"
            f"ANG_GT-{y_val[i]:.1f}_ANG_PRED-{y_pred_angle[i]:.1f}.png"
        )

        cv2.imwrite(
            os.path.join(output_dir, filename),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    X_val, y_val = load_validation_data(DATA_FOLDER, INPUT_SHAPE)

    MODEL_PATH = "./proposed_nn.h5"
    model = load_model(MODEL_PATH,compile=False)
    print(f"\n=== Avaliando modelo: {MODEL_PATH} ===")
    show_misclassified_images(
       model,
       X_val,
       y_val,
       CLASS_NAMES,
       max_images=20
    )

    MODEL_PATH = "./dave2_original.h5"
    model = load_model(MODEL_PATH,compile=False)
    print(f"\n=== Avaliando modelo: {MODEL_PATH} ===")
    show_misclassified_images(
       model,
       X_val,
       y_val,
       CLASS_NAMES,
       max_images=20
    )

