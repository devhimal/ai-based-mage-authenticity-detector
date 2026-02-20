# src/train.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


def train_model(
    img_size=(96, 96),
    batch_size=128,
    epochs=25,
    learning_rate=1e-4,
    patience=5,
    max_train_samples=None,
):
    """
    Trains the deepfake detection model.
    - Loads data from the `data/processed` directory.
    - Builds a model by fine-tuning a pre-trained MobileNetV2.
    - Trains the model and saves the best weights.
    - Saves a plot of the training history.
    """
    print("Starting model training...")

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    processed_data_dir = os.path.join(base_dir, "data", "processed")
    model_save_dir = os.path.join(base_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    # --- 1. Data Loading and Augmentation ---
    # Apply data augmentation to the training set
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Only rescale for validation and test sets
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(processed_data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",  # binary classification
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(processed_data_dir, "validation"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    # --- 2. Model Building (Transfer Learning) ---
    # Load the pre-trained base model (ResNet50V2) without the top classification layer
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3)
    )

    # Freeze the layers of the base model so they are not trained
    base_model.trainable = False

    # Add our custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)  # Dropout for regularization
    predictions = Dense(1, activation="sigmoid")(x)  # Sigmoid for binary classification

    model = Model(inputs=base_model.input, outputs=predictions)

    # --- 3. Model Compilation ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel Summary:")
    model.summary()

    # --- 4. Callbacks ---
    best_model_path = os.path.join(model_save_dir, "deepfake_detector_best.h5")

    # Save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    # Stop training early if there is no improvement
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    # --- 5. Model Training ---
    if max_train_samples:
        steps_per_epoch = min(train_generator.samples, max_train_samples) // batch_size
    else:
        steps_per_epoch = train_generator.samples // batch_size

    print("\nStarting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, early_stopping],
    )

    print("\nTraining complete.")

    # --- 6. Save Training History Plot ---
    history_plot_path = os.path.join(model_save_dir, "training_history.png")

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.suptitle("Model Training History")
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")


if __name__ == "__main__":
    train_model()

