# client/prepare_image_data.py

"""
Medical Image Dataset Preparation
====================================
Generates synthetic chest X-ray-like images for federated learning demo.

In a REAL hospital:
  - Images come from PACS (Picture Archiving and Communication System)
  - Format: DICOM (Digital Imaging and Communications in Medicine)
  - Types: Chest X-rays, CT scans, MRI
  - This script SIMULATES that process

We generate synthetic 64x64 grayscale images with patterns that
simulate different lung conditions:
  - Normal: Clear lung patterns
  - Pneumonia: Cloudy/opaque regions
  - COVID-19: Ground-glass opacity patterns

Dataset Reference:
  - NIH Chest X-ray Dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
  - CheXpert Dataset: https://stanfordmlgroup.github.io/competitions/chexpert/
  (These are too large for demo, so we simulate similar patterns)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import json


CLASS_NAMES = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
IMAGE_SIZE = 64


def generate_normal_xray(n_samples):
    """
    Generate synthetic normal chest X-ray images.
    Normal lungs: relatively clear, symmetric patterns
    """
    images = []
    for _ in range(n_samples):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        
        # Background lung field (lighter)
        img += np.random.uniform(0.3, 0.5)
        
        # Rib-like horizontal lines
        for y in range(10, IMAGE_SIZE, 8):
            thickness = np.random.randint(1, 3)
            img[y:y+thickness, 10:54] += np.random.uniform(0.1, 0.2)
        
        # Central mediastinum (darker vertical band)
        img[:, 28:36] += np.random.uniform(0.1, 0.2)
        
        # Clear lung fields (slightly darker on sides)
        img[:, :15] -= np.random.uniform(0.05, 0.1)
        img[:, 49:] -= np.random.uniform(0.05, 0.1)
        
        # Add slight noise
        img += np.random.normal(0, 0.03, (IMAGE_SIZE, IMAGE_SIZE))
        
        img = np.clip(img, 0, 1)
        images.append(img)
    
    return np.array(images)


def generate_pneumonia_xray(n_samples):
    """
    Generate synthetic pneumonia chest X-ray images.
    Pneumonia: cloudy/opaque consolidation regions in lungs
    """
    images = []
    for _ in range(n_samples):
        # Start with normal-ish base
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        img += np.random.uniform(0.3, 0.5)
        
        # Rib lines
        for y in range(10, IMAGE_SIZE, 8):
            img[y:y+2, 10:54] += np.random.uniform(0.1, 0.15)
        
        # Mediastinum
        img[:, 28:36] += np.random.uniform(0.1, 0.2)
        
        # PNEUMONIA: Add cloudy consolidation patches
        # Usually in one or both lower lobes
        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            cx = np.random.randint(15, 50)
            cy = np.random.randint(30, 55)  # Lower lung region
            radius = np.random.randint(5, 15)
            
            y_grid, x_grid = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2
            img[mask] += np.random.uniform(0.15, 0.35)
        
        # More noise than normal
        img += np.random.normal(0, 0.05, (IMAGE_SIZE, IMAGE_SIZE))
        
        img = np.clip(img, 0, 1)
        images.append(img)
    
    return np.array(images)


def generate_covid_xray(n_samples):
    """
    Generate synthetic COVID-19 chest X-ray images.
    COVID-19: bilateral ground-glass opacities, peripheral distribution
    """
    images = []
    for _ in range(n_samples):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        img += np.random.uniform(0.3, 0.5)
        
        # Rib lines (less visible due to opacities)
        for y in range(10, IMAGE_SIZE, 8):
            img[y:y+1, 10:54] += np.random.uniform(0.05, 0.1)
        
        # Mediastinum
        img[:, 28:36] += np.random.uniform(0.1, 0.15)
        
        # COVID: Ground-glass opacities (bilateral, peripheral)
        # Multiple small hazy patches on BOTH sides
        for side_offset in [0, 35]:  # Left and right lung
            num_patches = np.random.randint(3, 7)
            for _ in range(num_patches):
                cx = np.random.randint(8, 28) + side_offset
                cy = np.random.randint(15, 55)
                radius = np.random.randint(3, 10)
                
                y_grid, x_grid = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                distance = ((x_grid - cx)**2 + (y_grid - cy)**2).astype(float)
                # Soft edges (ground-glass appearance)
                mask = np.exp(-distance / (2 * radius**2))
                img += mask.astype(np.float32) * np.random.uniform(0.1, 0.25)
        
        # Slightly more hazy overall
        img += np.random.uniform(0.05, 0.1)
        img += np.random.normal(0, 0.04, (IMAGE_SIZE, IMAGE_SIZE))
        
        img = np.clip(img, 0, 1)
        images.append(img)
    
    return np.array(images)


def prepare_image_dataset(num_hospitals=3, samples_per_class=100):
    """
    Generate synthetic chest X-ray dataset and partition for hospitals.
    """
    print("\n" + "=" * 60)
    print("  🩻 MEDICAL IMAGE DATASET PREPARATION")
    print("  Type: Chest X-ray Classification")
    print("  Classes: Normal, Pneumonia, COVID-19")
    print("=" * 60)
    
    # Generate images for each class
    print(f"\n  📸 Generating {samples_per_class} images per class...")
    
    normal_imgs = generate_normal_xray(samples_per_class)
    print(f"  ✅ Normal X-rays: {len(normal_imgs)} images ({IMAGE_SIZE}x{IMAGE_SIZE})")
    
    pneumonia_imgs = generate_pneumonia_xray(samples_per_class)
    print(f"  ✅ Pneumonia X-rays: {len(pneumonia_imgs)} images")
    
    covid_imgs = generate_covid_xray(samples_per_class)
    print(f"  ✅ COVID-19 X-rays: {len(covid_imgs)} images")
    
    # Combine all
    all_images = np.concatenate([normal_imgs, pneumonia_imgs, covid_imgs])
    all_labels = np.concatenate([
        np.zeros(samples_per_class, dtype=np.int64),
        np.ones(samples_per_class, dtype=np.int64),
        np.full(samples_per_class, 2, dtype=np.int64)
    ])
    
    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    total = len(all_images)
    print(f"\n  📊 Total dataset: {total} images")
    print(f"  Normal: {(all_labels==0).sum()} | Pneumonia: {(all_labels==1).sum()} | COVID: {(all_labels==2).sum()}")
    
    # Partition for hospitals
    print(f"\n  🏥 Partitioning into {num_hospitals} hospitals...")
    
    split_size = total // num_hospitals
    
    for i in range(num_hospitals):
        hospital_name = f"hospital_{i+1}"
        hospital_dir = os.path.join("data", hospital_name, "images")
        os.makedirs(hospital_dir, exist_ok=True)
        
        start = i * split_size
        end = total if i == num_hospitals - 1 else start + split_size
        
        h_images = all_images[start:end]
        h_labels = all_labels[start:end]
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            h_images, h_labels, test_size=0.2, random_state=42, stratify=h_labels
        )
        
        # Reshape for PyTorch CNN: (N, 1, 64, 64) — 1 channel grayscale
        image_data = {
            "X_train": torch.FloatTensor(X_train).unsqueeze(1),  # Add channel dim
            "y_train": torch.LongTensor(y_train),
            "X_val": torch.FloatTensor(X_val).unsqueeze(1),
            "y_val": torch.LongTensor(y_val)
        }
        
        torch.save(image_data, os.path.join(hospital_dir, "image_data.pt"))
        
        # Save metadata
        metadata = {
            "hospital_name": hospital_name,
            "data_type": "Medical Imaging (Chest X-ray)",
            "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE} grayscale",
            "channels": 1,
            "total_images": len(h_images),
            "train_images": len(X_train),
            "val_images": len(X_val),
            "classes": CLASS_NAMES,
            "class_distribution": {
                "Normal": int((h_labels == 0).sum()),
                "Pneumonia": int((h_labels == 1).sum()),
                "COVID-19": int((h_labels == 2).sum())
            },
            "source": "Synthetic (simulating NIH Chest X-ray / CheXpert patterns)",
            "real_world_source": "Hospital PACS/DICOM system",
            "privacy_note": "Images remain on-premises. Only CNN weights are shared."
        }
        
        with open(os.path.join(hospital_dir, "image_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ {hospital_name}: {len(X_train)} train, {len(X_val)} val images")
        print(f"     Normal: {(h_labels==0).sum()} | Pneumonia: {(h_labels==1).sum()} | COVID: {(h_labels==2).sum()}")
    
    # Global test set
    test_dir = os.path.join("data", "test_set", "images")
    os.makedirs(test_dir, exist_ok=True)
    
    test_size = min(60, total // 5)
    test_images = all_images[:test_size]
    test_labels = all_labels[:test_size]
    
    test_data = {
        "X_test": torch.FloatTensor(test_images).unsqueeze(1),
        "y_test": torch.LongTensor(test_labels)
    }
    torch.save(test_data, os.path.join(test_dir, "image_test_data.pt"))
    
    print(f"\n  ✅ Test set: {test_size} images saved")
    
    print(f"\n  {'=' * 50}")
    print(f"  ✅ IMAGE DATASET PREPARATION COMPLETE!")
    print(f"  {'=' * 50}")
    print(f"""
  Files Created:
  ├── data/hospital_1/images/
  │   ├── image_data.pt           (PyTorch tensors: 1x64x64)
  │   └── image_metadata.json     (Class info, statistics)
  ├── data/hospital_2/images/
  ├── data/hospital_3/images/
  └── data/test_set/images/
      └── image_test_data.pt
  
  Model: CNN (Conv2d layers) — defined in server/image_model.py
  Classes: Normal (0) | Pneumonia (1) | COVID-19 (2)
  🔒 No images are shared — only CNN weights!
    """)


if __name__ == "__main__":
    prepare_image_dataset(num_hospitals=3, samples_per_class=100)