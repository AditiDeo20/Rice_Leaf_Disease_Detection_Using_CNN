"""
Inference module for Rice Leaf Disease Classification.
Provides preprocessing, prediction, probability calculation, and dual-subplot visualization.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

DEFAULT_CLASSES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]


def load_and_preprocess_image(image_path: str, target_size: tuple = (128, 128)) -> tuple:
    """
    1. Loads an image with cv2.imread and converts BGR to RGB.
    2. Resizes to target_size (128, 128) using bilinear/area interpolation.
    3. Normalizes pixel values to [0, 1] and adds batch dimension.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image at: {image_path}. Check format.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] and add batch dimension [1, 128, 128, 3]
    img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    return img_rgb, img_tensor


def predict_rice_disease(
    image_path: str,
    model,
    class_labels: list = DEFAULT_CLASSES,
    target_size: tuple = (128, 128),
    save_fig_path: str = None,
    show_plot: bool = True,
) -> dict:
    """
    Executes model inference to classify rice leaf disease:
    1. Loads an image with cv2.imread and converts BGR to RGB.
    2. Resizes to target_size (128, 128) and normalizes pixel values to [0, 1].
    3. Executes model inference to obtain class probabilities.
    4. Plots side-by-side: original leaf image and horizontal confidence bar chart.
    5. Returns dictionary with predicted class, confidence, and all class probabilities.
    """
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)

    img_rgb, img_tensor = load_and_preprocess_image(image_path, target_size=target_size)

    raw_preds = model.predict(img_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(raw_preds))
    predicted_class = class_labels[predicted_idx]
    predicted_confidence = float(raw_preds[predicted_idx] * 100.0)

    prob_dict = {
        cls: float(raw_preds[i] * 100.0) for i, cls in enumerate(class_labels)
    }

    # Plotting side-by-side: original leaf image and horizontal confidence bar chart
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot: original leaf image
    ax_img.imshow(img_rgb)
    ax_img.set_title(
        f"Query Leaf: {predicted_class}\nConfidence: {predicted_confidence:.2f}%",
        fontsize=12,
        fontweight="bold",
        color="darkgreen" if predicted_confidence > 70 else "darkorange",
    )
    ax_img.axis("off")

    # Right subplot: horizontal confidence bar chart
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    bars = ax_bar.barh(
        class_labels,
        [prob_dict[c] for c in class_labels],
        color=colors[: len(class_labels)],
        edgecolor="black",
        height=0.55,
    )
    ax_bar.set_xlim(0, 105)
    ax_bar.set_xlabel("Confidence (%)", fontsize=11, fontweight="bold")
    ax_bar.set_title("Disease Class Probabilities", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax_bar.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig_path)), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"Inference visualization saved to: {save_fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "predicted_class": predicted_class,
        "confidence": predicted_confidence,
        "probabilities": prob_dict,
        "image_path": image_path,
    }


# Alias for backward compatibility with previous requirements
predict_leaf_disease = predict_rice_disease


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rice Leaf Disease Inference Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to input rice leaf image")
    parser.add_argument("--model", type=str, default="models/rice_leaf_mobilenetv2.h5", help="Path to trained model")
    parser.add_argument("--output", type=str, default=None, help="Path to save output prediction plot")
    args = parser.parse_args()

    res = predict_rice_disease(
        image_path=args.image,
        model=args.model,
        class_labels=DEFAULT_CLASSES,
        save_fig_path=args.output,
        show_plot=True,
    )
    print(f"\nFinal Result: {res['predicted_class']} ({res['confidence']:.2f}%)")
