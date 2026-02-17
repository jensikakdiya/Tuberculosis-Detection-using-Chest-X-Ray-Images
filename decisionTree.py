import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (unchanged)
tb_path = r"TB_Chest_Radiography_Database\Tuberculosis"
normal_path = r"TB_Chest_Radiography_Database\Normal"

# Function to load dataset (UNCHANGED)
def load_dataset(normal_count, tb_count=700, img_size=(128, 128)):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore # import here so rest of script doesn't require TF
    tb_files = [f for f in os.listdir(tb_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    normal_files = [f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Always pick exactly tb_count images
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

# Train, evaluate and save using Decision Tree
def train_and_evaluate(normal_count, tb_count=700, random_state=42):
    print(f"\n==== Training with {normal_count} Normal images and {tb_count} TB images ====")
    X, y = load_dataset(normal_count, tb_count)

    # Flatten images to 2D feature vectors for Decision Tree: (n_samples, H*W*channels)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Train-Test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Build Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=random_state)

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    acc = clf.score(X_test, y_test)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Predictions and metrics
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "TB"]))
    print(f"\n==== Trained with {normal_count} Normal images and {tb_count} TB images ====")

    # Plot Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "TB"], yticklabels=["Normal", "TB"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Desicion Tree Confusion Matrix ({normal_count} Normal, {tb_count} TB)")
    #plt.show()
    plt.savefig(f"confusion_matrix_{normal_count}_normal_{tb_count}_tb.png")
    plt.close()
    print(f"💾 Confusion matrix saved as confusion_matrix_{normal_count}_normal_{tb_count}_tb.png")

    # Save the model (joblib)
    #model_filename = f"dt_model_normal{normal_count}_tb{tb_count}.joblib"
    #joblib.dump(clf, model_filename)
    #print(f"💾 Decision Tree model saved as {model_filename}")

    # Optionally save feature shape info so downstream code knows how to reshape inputs
    #meta_filename = f"dt_model_normal{normal_count}_tb{tb_count}_meta.npz"
    #np.savez(meta_filename, img_shape=X.shape[1:], flattened_dim=X_flat.shape[1])
    #print(f"💾 Model metadata saved as {meta_filename}")

# Run experiments (same numbers as original)
if __name__ == "__main__":
    # Set seeds for reproducibility of file selection
    random.seed(42)
    np.random.seed(42)

    train_and_evaluate(3500, 700, random_state=42)
    #train_and_evaluate(1400, 700, random_state=42)
    #train_and_evaluate(700, 700, random_state=42)