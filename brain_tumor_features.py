import os
import cv2
import pandas as pd
import numpy as np

# Load the existing brain_tumor_metadata.csv file
csv_path = "brain_tumor_metadata.csv"
df = pd.read_csv(csv_path)

# List to store processed data (with extracted features)
processed_data = []

# Loop through each row in the metadata CSV
for _, row in df.iterrows():
    img_path = row["Image_Path"]
    label = row["Label"]
    
    # Step 1: Convert image to grayscale using OpenCV
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    
    # Step 2: Resize image to a consistent size (e.g., 128x128 pixels)
    image_resized = cv2.resize(image, (128, 128))
    
    # Step 3: Extract features
    # - Mean pixel intensity
    mean_intensity = np.mean(image_resized)
    
    # - Texture (using Laplacian variance to capture texture details)
    texture = cv2.Laplacian(image_resized, cv2.CV_64F).var()
    
    # - Edge detection (Canny)
    edges = cv2.Canny(image_resized, 100, 200)
    edge_count = np.sum(edges == 255)  # Count white pixels in edges (edges detected)
    
    # Step 4: Store the processed data
    processed_data.append([img_path, label, mean_intensity, texture, edge_count])

# Create a DataFrame with metadata and features
processed_df = pd.DataFrame(processed_data, columns=["Image_Path", "Label", "Mean_Intensity", "Texture", "Edge_Count"])

# Save the preprocessed data to a new CSV file
preprocessed_csv_path = "brain_tumor_features.csv"
processed_df.to_csv(preprocessed_csv_path, index=False)

print(f"Preprocessed data saved successfully: {preprocessed_csv_path}")
