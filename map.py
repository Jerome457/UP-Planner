import os
import cv2
import numpy as np
import yaml

# ---------------------------
# CONFIG
# ---------------------------
map_file = os.path.join(os.path.dirname(__file__), "slam_map.pgm")
clean_map_file = os.path.join(os.path.dirname(__file__), "slam_map_border_cleaned.pgm")
yaml_file = os.path.join(os.path.dirname(__file__), "slam_map_border_cleaned.yaml")

map_resolution = 0.1  # meters per pixel

# Width of border to remove (meters)
BORDER_WIDTH_METERS = 2.0   # <-- increase this if needed
border_px = int(BORDER_WIDTH_METERS / map_resolution)

# ---------------------------
# LOAD MAP
# ---------------------------
img = cv2.imread(map_file, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Cannot load {map_file}")

if img.dtype != np.uint8:
    img = img.astype(np.uint8)

h, w = img.shape

# ---------------------------
# FILL UNKNOWN AS FREE (OPTIONAL)
# ---------------------------
# ROS convention: unknown = 205
img[img == 205] = 254

# ---------------------------
# REMOVE WIDE BLACK BORDERS
# ---------------------------
# Top
img[0:border_px, :] = 254
# Bottom
img[h-border_px:h, :] = 254
# Left
img[:, 0:border_px] = 254
# Right
img[:, w-border_px:w] = 254

# ---------------------------
# SAVE CLEANED MAP
# ---------------------------
cv2.imwrite(clean_map_file, img)

yaml_data = {
    'image': os.path.basename(clean_map_file),
    'resolution': map_resolution,
    'origin': [0.0, 0.0, 0.0],
    'negate': 0,
    'occupied_thresh': 0.65,
    'free_thresh': 0.196
}

with open(yaml_file, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"Cleaned map saved: {clean_map_file}")
print(f"YAML saved: {yaml_file}")
print(f"Removed border width: {border_px} pixels ({BORDER_WIDTH_METERS} m)")
