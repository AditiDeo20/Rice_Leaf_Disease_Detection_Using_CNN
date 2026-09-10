"""
Model architecture definitions for Rice Leaf Disease Classification.
Includes Custom Progressive CNN and MobileNetV2 Transfer Learning with fine-tuning capabilities.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


def build_custom_cnn(
    input_shape: tuple = (128, 128, 3),
    num_classes: int = 3,
    learning_rate: float = 1e-4,
) -> Sequential:
    """
    Builds and compiles a progressive Custom Convolutional Neural Network.
    Architecture:
      - Block 1: Conv2D(32, 3x3) -> BatchNorm -> MaxPool2D(2x2)
      - Block 2: Conv2D(64, 3x3) -> BatchNorm -> MaxPool2D(2x2)
      - Block 3: Conv2D(128, 3x3) -> BatchNorm -> MaxPool2D(2x2)
      - Classifier: Flatten -> Dense(128, relu) -> Dropout(0.5) -> Dense(num_classes, softmax)
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ], name="custom_progressive_cnn")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2(
    input_shape: tuple = (128, 128, 3),
    num_classes: int = 3,
    learning_rate: float = 1e-4,
    freeze_base: bool = True,
) -> Model:
    """
    Constructs a Transfer Learning model using pre-trained MobileNetV2.
    The base feature extractor is initially frozen to preserve ImageNet representations.
    A top classification head with GlobalAveragePooling, Dense, BatchNorm, and Dropout(0.4) is added.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = not freeze_base

    inputs = Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dense(128, activation="relu", name="head_dense")(x)
    x = BatchNormalization(name="head_bn")(x)
    x = Dropout(0.4, name="head_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer_learning")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_fine_tuning(
    model: Model,
    fine_tune_layers: int = 30,
    learning_rate: float = 1e-5,
) -> Model:
    """
    Unfreezes the top N layers of the base MobileNetV2 model for fine-tuning.
    Re-compiles the model with a conservative, low learning rate (1e-5).
    """
    # Locate the MobileNetV2 base model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find base MobileNetV2 model inside the architecture.")

    base_model.trainable = True

    # Freeze all layers before the top fine_tune_layers
    num_layers = len(base_model.layers)
    cutoff = max(0, num_layers - fine_tune_layers)
    for layer in base_model.layers[:cutoff]:
        layer.trainable = False
    for layer in base_model.layers[cutoff:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Base model unfrozen for fine-tuning: {fine_tune_layers} layers trainable (cutoff: {cutoff}/{num_layers})")
    return model
