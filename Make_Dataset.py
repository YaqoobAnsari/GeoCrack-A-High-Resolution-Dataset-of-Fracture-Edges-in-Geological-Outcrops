import os
import numpy as np
from PIL import Image
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# Define folder paths
folder1_path = 'C:/Users/User/Desktop/Edges/Original Images'
folder2_path = 'C:/Users/User/Desktop/Edges/Edge Binary Masks'
patched_folder_path = 'C:/Users/User/Desktop/Edges/patched_images'
overlay_folder_path = 'C:/Users/User/Desktop/Edges/overlay_images'
os.makedirs(patched_folder_path, exist_ok=True)
os.makedirs(overlay_folder_path, exist_ok=True)

# Convert non-PNG images to PNG and remove original files
def convert_images_to_png(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                png_filename = f"{os.path.splitext(filename)[0]}.png"
                png_path = os.path.join(folder_path, png_filename)
                img.save(png_path)
            os.remove(file_path)
            print(f"Converted {filename} to {png_filename} and removed the original file.")

# Ensure each image in Original Images has a corresponding image in Edge Binary Mask and both have the same dimensions
def validate_image_pairs(folder1_path, folder2_path):
    folder1_images = os.listdir(folder1_path)
    folder2_images = os.listdir(folder2_path)
    valid_pairs = []

    for image_name in folder1_images:
        if image_name in folder2_images:
            image1 = Image.open(os.path.join(folder1_path, image_name))
            image2 = Image.open(os.path.join(folder2_path, image_name))
            if image1.size == image2.size:
                valid_pairs.append(image_name)
    print(f"Found {len(valid_pairs)} valid image pairs.")
    return valid_pairs

# Convert images in Edge Binary Mask to binary masks
def convert_to_binary_mask(image):
    image_array = np.array(image.convert('L'))  # Ensure it's single-channel
    binary_mask = (image_array > 0).astype(np.uint8) * 255
    return Image.fromarray(binary_mask)

# Create overlay image 
def create_overlay(original_patch, binary_mask_patch):
    original_array = np.array(original_patch)
    binary_mask_array = np.array(binary_mask_patch)
    overlay = original_array.copy()
    red_channel = overlay[:, :, 0]
    red_channel[binary_mask_array > 0] = 255
    overlay[:, :, 0] = red_channel
    return Image.fromarray(overlay)

# Create patches of 224x224 and save valid pairs
def create_patches(image_pairs, folder1_path, folder2_path, patched_folder_path, overlay_folder_path, patch_size=224, edge_threshold=0.01, min_white_pixel_ratio=0.01):
    csv_file_path = 'C:/Users/User/Desktop/Edges/patch_pairs.csv'
    total_patches = 0
    saved_patches = 0
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Original Image Patch', 'Binary Mask Image Patch'])

        for image_name in image_pairs:
            original_image = Image.open(os.path.join(folder1_path, image_name))
            binary_mask_image = convert_to_binary_mask(Image.open(os.path.join(folder2_path, image_name)))

            original_array = np.array(original_image)
            binary_mask_array = np.array(binary_mask_image)

            height, width = original_array.shape[:2]
            patch_number = 0  # Reset patch_number for each image pair

            # Initialize counters for each image
            image_total_patches = 0
            image_saved_patches = 0

            for i in range(0, height, patch_size):
                for j in range(0, width, patch_size):
                    image_total_patches += 1
                    original_patch = original_array[i:i+patch_size, j:j+patch_size]
                    binary_mask_patch = binary_mask_array[i:i+patch_size, j:j+patch_size]

                    # Skip non-square patches and patches with no or almost no white pixels
                    if original_patch.shape[0] != patch_size or original_patch.shape[1] != patch_size:
                        continue
                    edge_pixel_count = np.sum(binary_mask_patch > 0)
                    total_pixel_count = patch_size * patch_size
                    if edge_pixel_count / total_pixel_count < min_white_pixel_ratio:  # Ensure substantial amount of white pixels
                        continue

                    original_patch_img = Image.fromarray(original_patch)
                    binary_mask_patch_img = Image.fromarray(binary_mask_patch)

                    original_patch_name = f"{image_name.split('.')[0]}_original_patch{patch_number}.png"
                    binary_mask_patch_name = f"{image_name.split('.')[0]}_binarymask_patch{patch_number}.png"

                    original_patch_img.save(os.path.join(patched_folder_path, original_patch_name))
                    binary_mask_patch_img.save(os.path.join(patched_folder_path, binary_mask_patch_name))

                    # Create and save overlay image
                    overlay_img = create_overlay(original_patch_img, binary_mask_patch_img)
                    overlay_name = f"{image_name.split('.')[0]}_overlay_patch{patch_number}.png"
                    overlay_img.save(os.path.join(overlay_folder_path, overlay_name))

                    # Write to CSV
                    csv_writer.writerow([os.path.join(patched_folder_path, original_patch_name), os.path.join(patched_folder_path, binary_mask_patch_name)])

                    patch_number += 1
                    image_saved_patches += 1

            print(f"Processed {image_name}: made {image_total_patches} patches, retained {image_saved_patches} patches.")
            total_patches += image_total_patches
            saved_patches += image_saved_patches

    print(f"Total patches processed: {total_patches}")
    print(f"Total patches saved: {saved_patches}")

# Process the images
print("Converting non-PNG images to PNG...")
convert_images_to_png(folder1_path)
convert_images_to_png(folder2_path)
print("All files are PNG")
print("Validating image pairs...")
valid_image_pairs = validate_image_pairs(folder1_path, folder2_path)
print("Creating patches...")
create_patches(valid_image_pairs, folder1_path, folder2_path, patched_folder_path, overlay_folder_path)

# Load the patch pairs CSV file
csv_file_path = 'C:/Users/User/Desktop/Edges/patch_pairs.csv'
patch_pairs_df = pd.read_csv(csv_file_path)

# Shuffle the dataframe
print("Shuffling data...")
patch_pairs_df = patch_pairs_df.sample(frac=1).reset_index(drop=True)

# Calculate the number of samples for each split
total_samples = len(patch_pairs_df)
train_size = int(total_samples * 0.5)
val_test_size = total_samples - train_size
val_size = int(val_test_size * 0.5)
test_size = val_test_size - val_size

# Split the dataframe
print("Splitting data into train, validation, and test sets...")
train_df, val_test_df = train_test_split(patch_pairs_df, train_size=train_size, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=test_size, random_state=42)

# Save the splits into separate CSV files
train_csv_path = 'C:/Users/User/Desktop/Edges/train.csv'
val_csv_path = 'C:/Users/User/Desktop/Edges/validation.csv'
test_csv_path = 'C:/Users/User/Desktop/Edges/test.csv'

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

print("All tasks completed successfully.")
