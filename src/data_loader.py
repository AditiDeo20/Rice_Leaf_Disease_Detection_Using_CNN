"""
Data loading and preprocessing module for Rice Leaf Disease Classification.
Handles dataset partitioning, anti-leakage augmentation, and generator instantiation.
"""

import os
import shutil
import random
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TARGET_CLASSES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]


def prepare_dataset(
    source_dir: str,
    output_dir: str = "./dataset_split",
    split_ratio: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """
    Splits the raw dataset into Train, Validation, and Test sets.
    Applies splitting BEFORE any augmentation to avoid data leakage.

    Uses `splitfolders` if installed, or falls back to a deterministic
    stratified split using standard library shutil and random.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    try:
        import splitfolders

        splitfolders.ratio(
            source_dir,
            output=output_dir,
            seed=seed,
            ratio=split_ratio,
        )
    except ImportError:
        # Robust fallback if splitfolders is not installed
        rng = random.Random(seed)
        for cls in TARGET_CLASSES:
            src_cls_dir = os.path.join(source_dir, cls)
            if not os.path.exists(src_cls_dir):
                continue
            images = [
                f
                for f in os.listdir(src_cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            rng.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * split_ratio[0])
            n_val = int(n_total * split_ratio[1])

            train_imgs = images[:n_train]
            val_imgs = images[n_train : n_train + n_val]
            test_imgs = images[n_train + n_val :]

            for split_name, split_imgs, target_base in [
                ("train", train_imgs, train_dir),
                ("val", val_imgs, val_dir),
                ("test", test_imgs, test_dir),
            ]:
                dest_folder = os.path.join(target_base, cls)
                os.makedirs(dest_folder, exist_ok=True)
                for img_name in split_imgs:
                    shutil.copy2(
                        os.path.join(src_cls_dir, img_name),
                        os.path.join(dest_folder, img_name),
                    )

    # Verification report
    print(f"Dataset split completed successfully into: {output_dir}")
    counts = {}
    for split_name, split_path in [
        ("train", train_dir),
        ("val", val_dir),
        ("test", test_dir),
    ]:
        counts[split_name] = {}
        for cls in TARGET_CLASSES:
            folder = os.path.join(split_path, cls)
            if os.path.exists(folder):
                counts[split_name][cls] = len(
                    [
                        f
                        for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
        print(f" - {split_name.capitalize()} counts: {counts[split_name]}")

    return {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir,
        "counts": counts,
    }


def create_generators(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    target_size: tuple = (128, 128),
    batch_size: int = 16,
    seed: int = 42,
):
    """
    Creates Keras ImageDataGenerators.
    Augmentation is applied exclusively to the training set.
    Validation and test sets are rescaled only (1/255) to prevent distribution contamination.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.20,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_labels = list(train_gen.class_indices.keys())
    return train_gen, val_gen, test_gen, class_labels


def load_flattened_data(
    directory: str,
    target_classes: list = TARGET_CLASSES,
    img_size: tuple = (64, 64),
):
    """
    Loads and flattens images to 1D vectors for classical ML baseline models (SVM, Random Forest).
    Pixel values are normalized to the [0, 1] range.
    """
    X, y = [], []
    class_map = {name: idx for idx, name in enumerate(target_classes)}

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    for cls_name, cls_idx in class_map.items():
        folder = os.path.join(directory, cls_name)
        if not os.path.exists(folder):
            continue
        for img_name in sorted(os.listdir(folder)):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    X.append(img.flatten() / 255.0)
                    y.append(cls_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
