import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore


def create_cnn_model(input_shape, num_classes=3, learning_rate=1e-3):
    """
    Build and compile a 1D CNN model for financial time-series classification.

    Args:
        input_shape (tuple): Shape of input data (timesteps, features).
        num_classes (int): Number of output classes (e.g., 3 for long/hold/short).
        learning_rate (float): Initial learning rate for Adam optimizer.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Convolutional feature extractor ---
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # --- Fully connected head ---
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # --- Output layer ---
    if num_classes == 1:
        outputs = layers.Dense(1, activation='tanh')(x)
        loss_fn = 'mse'
        metrics = ['mae']
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss_fn = 'sparce_categorical_crossentropy'
        metrics = ['accuracy']

    model = models.Model(inputs, outputs, name="cnn_trading_model")

    # --- Compile ---
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])

    return model


def summary(model):
    """Convenience function to print model summary."""
    model.summary()


if __name__ == "__main__":
    # Example quick test
    test_model = create_cnn_model(input_shape=(60, 20), num_classes=3)
    test_model.summary()