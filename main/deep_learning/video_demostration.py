import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "./deeppicar_pretrained_aug.h5"
DATA_FOLDER = "./deep_learning/resized_img_road"
CSV_NAME = "labels_converted.csv"

INPUT_SHAPE = (200, 66, 3)   # (H, W, C)
VIDEO_OUT = "predicted_angles_video.mp4"
FPS = 10

model = load_model(
    "./deeppicar_pretrained_aug.h5",
    custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
)

def dataset_iterator(data_folder, input_shape):
    csv_path = os.path.join(data_folder, CSV_NAME)

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            img_name, gt_angle = row[0], float(row[1])
            img_path = os.path.join(data_folder, img_name)

            if not os.path.exists(img_path):
                continue

            img_pil = Image.open(img_path).convert("RGB")
            img_pil = img_pil.resize((input_shape[1], input_shape[0]))

            img_np = np.asarray(img_pil, dtype=np.float32) / 255.0

            yield img_name, img_np, gt_angle

def annotate_image(image, pred_angle, gt_angle=None):
    """
    image: RGB float [0,1]
    retorna imagem BGR uint8 anotada
    """
    img = (image * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Texto compacto
    text_pred = f"Pred: {pred_angle:.1f} deg"

    if gt_angle is not None:
        text_gt = f"GT: {gt_angle:.1f} deg"
    else:
        text_gt = None

    # Overlay mínimo (altura ~18 px)
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)

    # Fonte pequena e fina
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    thickness = 1

    # Predição (linha superior)
    cv2.putText(
        img,
        text_pred,
        (3, 12),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    # Ground truth (opcional, abaixo)
    if text_gt:
        cv2.putText(
            img,
            text_gt,
            (110, 12),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA
        )

    return img

def generate_prediction_video(
    model,
    data_folder,
    input_shape,
    output_video,
    fps=10
):
    iterator = dataset_iterator(data_folder, input_shape)

    # Inicializa com a primeira imagem para pegar dimensões
    first_name, first_img, first_gt = next(iterator)

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Processa a primeira
    pred = model.predict(first_img[None, ...], verbose=0)[0, 0]
    annotated = annotate_image(first_img, pred, first_gt)
    writer.write(annotated)

    count = 1

    # Loop principal
    for img_name, img, gt_angle in iterator:
        pred_angle = model.predict(img[None, ...], verbose=0)[0, 0]
        annotated = annotate_image(img, pred_angle, gt_angle)
        writer.write(annotated)
        count += 1

        if count % 100 == 0:
            print(f"{count} frames processados...")

    writer.release()
    print(f"Vídeo gerado com sucesso: {output_video}")
    print(f"Total de frames: {count}")

if __name__ == "__main__":
    generate_prediction_video(
        model=model,
        data_folder=DATA_FOLDER,
        input_shape=INPUT_SHAPE,
        output_video=VIDEO_OUT,
        fps=FPS
    )
