import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Paths (same as your CNN & DT models)
tb_path = r"TB_Chest_Radiography_Database\Tuberculosis"
normal_path = r"TB_Chest_Radiography_Database\Normal"


# Function to load dataset (same logic as other scripts)
def load_dataset(normal_count, tb_count=700, img_size=(128, 128)):
    tb_files = [f for f in os.listdir(tb_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    normal_files = [f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Random selection
    tb_selected = random.sample(tb_files, tb_count)
    normal_selected = random.sample(normal_files, normal_count)

    X, y = [], []

    for f in tb_selected:
        img = load_img(os.path.join(tb_path, f), target_size=img_size, color_mode="grayscale")
        X.append(img_to_array(img))
        y.append(1)  # TB = 1

    for f in normal_selected:
        img = load_img(os.path.join(normal_path, f), target_size=img_size, color_mode="grayscale")
        X.append(img_to_array(img))
        y.append(0)  # Normal = 0

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)

    return X, y


# ANN Model (Dense layers only)
def build_ann_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),   # Flatten images → vector
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')   # Binary output
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Train & Evaluate
def train_and_evaluate(normal_count, tb_count=700, epochs=10, batch_size=32):
    print(f"\n==== Training ANN with {normal_count} Normal images and {tb_count} TB images ====")

    X, y = load_dataset(normal_count, tb_count)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build ANN model
    model = build_ann_model(X_train.shape[1:])
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n",
          classification_report(y_test, y_pred, target_names=["Normal", "TB"]))
    print(f"\n==== Trained with {normal_count} Normal images and {tb_count} TB images ====")

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "TB"], yticklabels=["Normal", "TB"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ANN Confusion Matrix")
    #plt.show()
    plt.savefig(f"ann_confusion_matrix_{normal_count}_normal_{tb_count}_tb.png")
    plt.close()

    print(f"💾 Confusion matrix saved as ann_confusion_matrix_{normal_count}_normal_{tb_count}_tb.png")

    # Save model
    # model.save(f"ann_model_normal{normal_count}_tb{tb_count}.h5")


# Run ANN experiment
if __name__ == "__main__":
    train_and_evaluate(3500, 700, epochs=10)
    #train_and_evaluate(1400, 700, epochs=10)
    #train_and_evaluate(700, 700, epochs=10)
