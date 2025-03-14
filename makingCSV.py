import os
import pandas as pd

# ✅ Correct dataset path
dataset_path = "D:/Github/Projects/ChetanSir/AIML/Training"

# Define categories (folder names = labels)
categories = ["pituitary", "notumor", "meningioma", "glioma"]

# List to store metadata
data = []

# Loop through each category folder
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    
    # ✅ Check if folder exists before reading
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder not found - {folder_path}")
        continue
    
    # Get all image filenames in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):  # Check valid image formats
            img_path = os.path.join(folder_path, filename)  # Full image path
            data.append([img_path, category])  # Append path and label

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=["Image_Path", "Label"])

# Save to CSV
csv_path = "brain_tumor_metadata.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully: {csv_path}")  # ✅ Removed Unicode emoji for Windows compatibility
