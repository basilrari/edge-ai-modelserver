import cv2
import numpy as np

def divide_into_grid(image, rows=3, cols=3):
    height, width = image.shape[:2]
    grid_h, grid_w = height // rows, width // cols
    grids = []

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * grid_w, r * grid_h
            x2, y2 = (c + 1) * grid_w, (r + 1) * grid_h
            grid = image[y1:y2, x1:x2]
            grids.append(((x1, y1, x2, y2), grid))
    
    return grids

def analyze_grid(mask, threshold=0.1, rows=3, cols=3):
    grid_info = []
    grids = divide_into_grid(mask, rows, cols)

    for idx, ((x1, y1, x2, y2), grid) in enumerate(grids):
        flooded_pixels = np.sum(grid > 0)
        total_pixels = grid.size
        flood_ratio = flooded_pixels / total_pixels
        grid_info.append({
            "id": idx + 1,
            "coordinates": (x1, y1, x2, y2),
            "flooded": flood_ratio > threshold,
            "ratio": flood_ratio
        })
    
    return grid_info

def draw_grid(image, grid_info):
    output = image.copy()

    for grid in grid_info:
        x1, y1, x2, y2 = grid["coordinates"]
        color = (0, 0, 255) if grid["flooded"] else (0, 255, 0)
        label = f'{grid["ratio"]*100:.1f}%'
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return output
