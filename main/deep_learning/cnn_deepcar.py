import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random
import csv

# =============================
# 1) Modelo (PilotNet / DeepPiCar)
# =============================
def build_model(input_shape):
    drop_rate = .6
    model = keras.Sequential([
        layers.Conv2D(24, (5,5), strides=(2,2), input_shape=input_shape),
        layers.ELU(),

        layers.Conv2D(36, (5,5), strides=(2,2)),
        layers.ELU(),

        layers.Conv2D(48, (5,5), strides=(2,2)),
        layers.ELU(),

        layers.Conv2D(64, (3,3)),
        layers.ELU(),

        layers.Conv2D(64, (3,3)),
        layers.ELU(),

        layers.Flatten(),

        layers.Dense(512,  kernel_regularizer=keras.regularizers.l1(0.001)),  # Aumente para potência de 2
        layers.BatchNormalization(),  # Estabiliza
        layers.Dropout(drop_rate),  # Reduza dropout
        layers.ELU(),
        
        layers.Dense(128,  kernel_regularizer=keras.regularizers.l1(0.001)),  # Segunda camada também maior
        layers.BatchNormalization(),  # Estabiliza
        layers.Dropout(drop_rate),  # Reduza dropout
        layers.ELU(),
        
        # Saída direta
        layers.Dense(1, activation='linear')
    ])
    return model



# =============================
# 2) Função de Augmentation leve
# =============================
def augment_image(img):

    """Aplica pequena rotação (< ±10°), shift leve (~5%) e variação de brilho."""
    angle = random.uniform(-5, 5)
    shift_x = random.uniform(-0.05, 0.05) * img.width
    shift_y = random.uniform(-0.05, 0.05) * img.height

    # Rotação
    aug = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Shift
    aug = aug.transform(
        aug.size,
        Image.AFFINE,
        (1, 0, shift_x, 0, 1, shift_y),
        resample=Image.BICUBIC
    )

    enhancer = ImageEnhance.Brightness(aug)
    aug = enhancer.enhance(random.uniform(0.9, 1.1))

    return aug


# =============================
# 3) Carregar dados + Oversample RIGHT (30°)
# =============================
def load_data(data_folder, input_shape, oversample_factor=3):

    """
    Lê imagens e rótulos de labels.csv e faz oversampling da classe 30° (right turn).
    """
    csv_path = os.path.join(data_folder, "labels_converted.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"labels_converted.csv não encontrado em {data_folder}")

    images, labels = [], []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            img_name, label = row[0], float(row[1])
            img_path = os.path.join(data_folder, img_name)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            img = img.resize((input_shape[1], input_shape[0]))  # (width, height)
            img = np.asarray(img, dtype=np.float32) / 255.0

            images.append(img)
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    # Oversampling manual (fator 2 para 0° e 30°)
    X_aug, y_aug = list(X), list(y)
    for i in range(len(y)):
        if y[i] == 0 or y[i] == 30:  # retas e curvas à direita
            X_aug.append(X[i])
            y_aug.append(y[i])
        if y[i] == 30:  # curvas à direita
            X_aug.append(X[i])
            y_aug.append(y[i])
        

    X = np.array(X_aug)
    y = np.array(y_aug)

    print("Contagem após oversampling:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u:>5}° : {c} amostras")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Total: {len(X)} imagens -> Treino: {len(X_train)}, Val: {len(X_val)}, Teste: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# =============================
# 4) Matriz de confusão (salva)
# =============================
def save_confusion_matrix(y_true, y_pred_labels, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred_labels)
    print("Classification Report:\n", classification_report(y_true, y_pred_labels, target_names=class_names))

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")


# =============================
# 5) Treinar e avaliar
# =============================
def train_and_evaluate(data_folder, save_model_path, cm_save_path, input_shape, epochs=20, batch_size=32):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_folder, input_shape, oversample_factor=3)

    model = build_model(input_shape)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='mse',
                  metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    plot_training_history(
    history,
    save_path="./training_vs_validation_loss_proposed_nn_l2.png"
)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    model.save(save_model_path)
    print(f"Modelo salvo em: {save_model_path}")

    y_pred = model.predict(X_test).flatten()

    def to_class(angle, thr=5.0):
        if angle < -thr:
            return 0 
        elif angle > thr:
            return 2 
        else:
            return 1 

    y_true_cls = [to_class(v) for v in y_test]
    y_pred_cls = [to_class(v) for v in y_pred]

    save_confusion_matrix(y_true_cls, y_pred_cls, ["Left", "Center", "Right"], cm_save_path)

    return model, history

def plot_training_history(history, save_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.xlabel('Época')
    plt.ylabel('MSE Loss')
    plt.title('Perda de Treinamento vs Validação')
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

# =============================
# 6) Execução principal
# =============================

if __name__ == "__main__":
    data_folder = "./deep_learning/resized_img_road"  # pasta contendo imagens + labels
    save_model_path = "./proposed_nn_l2.h5"
    input_shape = (200, 66, 3)  # formato usado pelo DeepPiCar (altura, largura, canais)
    cm_save_path = "./proposed_nn_l2.png"

    train_and_evaluate(
        data_folder,
        save_model_path,
        cm_save_path,
        input_shape,
        epochs=20,
        batch_size=32
    )
