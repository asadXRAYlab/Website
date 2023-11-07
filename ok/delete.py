import json
import numpy as np
import cv2

# Load the JSON data
data = [
    {
        "format": "rle",
        "rle": [0, 14, 16, 0, 57, 27, 255, 255, 255, 0, 233, 163, 224, 26, 143, 255, 132, 155, 128, 112, 18, 127, 254, 17, 78, 1, 192, 104, 255, 248, 66, 56, 7, 1, 231, 255, 225, 0, 224, 28, 8, 159, 255, 131, 233, 128, 112, 36, 127, 254, 15, 86, 1, 192, 151, 255, 291, 255, 224, 173, 224, 25, 223, 255, 103, 128, 112, 46, 63, 254, 11, 174, 1, 156, 255, 247, 152, 7, 2, 151, 255, 224, 192, 224, 25, 159, 255, 128, 149, 128, 112, 32, 63, 254, 12, 246, 1, 145, 255, 248, 11, 184, 7, 1, 35, 255, 140, 2, 255, 254, 31, 0, 143, 255, 131, 125, 128, 95, 255, 225, 192, 96, 23, 255, 248, 79, 152, 4, 127, 255, 255, 252, 3, 255, 255, 128, 127, 255, 240, 15, 13, 22, 0],
        "brushlabels": ["car"],
        "original_width": 640,
        "original_height": 360,
    }
]

# Extract RLE data
rle_data = data[0]["rle"]
width = data[0]["original_width"]
height = data[0]["original_height"]

# Decode the RLE data into a binary mask
mask = np.zeros((height, width), dtype=np.uint8)
current_pixel = 0
for i in range(len(rle_data) // 2):
    run_length = rle_data[2 * i]
    value = rle_data[2 * i + 1]
    mask[current_pixel : current_pixel + run_length] = value
    current_pixel += run_length

# Find object contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Write YOLO format to a text file
with open("yolo_format.txt", "w") as file:
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        width_normalized = w / width
        height_normalized = h / height
        class_index = 0  # Assuming a single class with index 0

        line = f"{class_index} {x_center} {y_center} {width_normalized} {height_normalized}\n"
        file.write(line)
