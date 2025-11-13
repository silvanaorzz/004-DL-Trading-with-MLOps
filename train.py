"""
train.py
--------
Training pipeline for the CNN-based trading model.
Includes class weighting, MLFlow experiment tracking, artifact saving
for consistent inference (scaler + label mapping).
"""

import os
import joblib
import numpy as np
import mlflow
import mlflow.tensorflow
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt


def compute_class_weights(y):
    """Compute class weights for imbalanced datasets."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def plot_training_history(history, save_path=None):
    """Plot and optionally save training/validation loss and accuracy curves."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title('Accuracy')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_training_artifacts(X_train, y_train, model_dir="models"):
    """
    Save the scaler and label mapping used for training.
    These artifacts ensure the FastAPI model reproduces preprocessing exactly.
    """
    os.makedirs(model_dir, exist_ok=True)

    # --- Save normalization scaler ---
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # --- Save label mapping ---
    unique_labels = sorted(list(np.unique(y_train)))
    label_mapping_path = os.path.join(model_dir, "label_mapping.joblib")
    joblib.dump(unique_labels, label_mapping_path)

    print(f"✅ Saved scaler to {scaler_path}")
    print(f"✅ Saved label mapping to {label_mapping_path}")

    return scaler, unique_labels


def train_model(model, X_train, y_train, X_val, y_val, experiment_name="CNN_Trading"):
    """
    Train a CNN model using TensorFlow/Keras and log everything to MLFlow.

    Args:
        model: Keras model instance.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        experiment_name (str): Name of the MLFlow experiment.

    Returns:
        model: Trained model.
    """
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # --- Save scaler and label mapping ---
    scaler, label_mapping = save_training_artifacts(X_train.reshape(X_train.shape[0], -1), y_train)

    # --- Prepare MLFlow ---
    mlflow.set_experiment(experiment_name)

    # --- Handle class weights ---
    class_weights = compute_class_weights(y_train)

    # --- Convert labels for multi-class ---
    num_classes = len(np.unique(y_train))
    if num_classes > 2:
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    else:
        y_train_cat, y_val_cat = y_train, y_val

    # --- Define callbacks ---
    checkpoint_path = "models/best_cnn_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    with mlflow.start_run(run_name="cnn_training"):
        # Log hyperparameters
        mlflow.log_params({
            "optimizer": "Adam",
            "learning_rate": model.optimizer.learning_rate.numpy(),
            "batch_size": 64,
            "epochs": 50,
            "architecture": "1D CNN",
            "num_classes": num_classes
        })

        # --- Train model ---
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=50,
            batch_size=64,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Generate predictions for report
        y_val_pred = model.predict(X_val)

        # 🩹 Fix: convert probabilities to class labels
        if y_val_pred.ndim > 1:
            y_val_pred = np.argmax(y_val_pred, axis=1)

        report = classification_report(y_val, y_val_pred, output_dict=True)
        print("Classification Report:")
        print(pd.DataFrame(report).transpose())

        # --- Save trained model ---
        model.save(checkpoint_path)
        mlflow.tensorflow.log_model(model=model, artifact_path="cnn_model")

        # --- Plot training curves ---
        plot_path = "artifacts/training_history.png"
        plot_training_history(history, save_path=plot_path)
        mlflow.log_artifact(plot_path)

        # --- Evaluate on validation set ---
        y_val_pred = np.argmax(model.predict(X_val), axis=1)
        if y_val_pred.ndim > 1:
            y_val_pred = np.argmax(y_val_pred, axis=1)

        report = classification_report(y_val, y_val_pred, output_dict=True)
        conf_mat = confusion_matrix(y_val, y_val_pred)

        # --- Log metrics ---
        mlflow.log_metrics({
            "val_accuracy": report["accuracy"],
            "val_precision": report["weighted avg"]["precision"],
            "val_recall": report["weighted avg"]["recall"],
            "val_f1": report["weighted avg"]["f1-score"]
        })

        # --- Log confusion matrix ---
        conf_path = "artifacts/confusion_matrix.png"
        plt.figure(figsize=(5, 4))
        plt.imshow(conf_mat, cmap="Blues")
        plt.title("Validation Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(conf_path)
        plt.close()
        mlflow.log_artifact(conf_path)

        print("✅ Model training completed and logged to MLFlow.")
        return model
