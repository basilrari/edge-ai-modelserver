import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.models.segmentation as models
from PIL import Image as PILImage
from geographiclib.geodesic import Geodesic
import time
import csv
from collections import deque


class DeepLabInference:
    def __init__(self, csv_file='latency_log.csv', rolling_window=30):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # Load DeepLabV3 model
        self.model = models.deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)
        self.model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Camera intrinsics
        self.fx = 277.19
        self.fy = 277.19
        self.cx = 160.5
        self.cy = 120.5
        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(K)

        # Geodesic and rotation for GPS calculation
        self.geod = Geodesic.WGS84
        theta = np.deg2rad(45)
        self.R_pitch = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Dummy GPS and altitude
        self.current_gps = (37.7749, -122.4194)
        self.drone_altitude = 3.0

        # CSV and rolling buffers
        self.csv_file = csv_file
        self.rolling_window = rolling_window
        self.latency_buffers = {
            'image': deque(maxlen=rolling_window),
            'ai': deque(maxlen=rolling_window),
            'gps': deque(maxlen=rolling_window),
            'total': deque(maxlen=rolling_window)
        }
        self.fps_buffer = deque(maxlen=rolling_window)

        # Initialize CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'ImageProcessing_ms', 'AIInference_ms', 'GPSConversion_ms', 'Total_ms'])

    def process_frame(self, frame, frame_num, dt):
        t0 = time.time()
        # IMAGE PROCESSING
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0)
        t1 = time.time()

        # AI INFERENCE
        with torch.no_grad():
            output = self.model(img_tensor)['out']
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        t2 = time.time()

        # Resize mask and overlay
        mask_resized = cv2.resize(pred.astype(np.uint8), (frame.shape[1], frame.shape[0])) * 255
        mask_resized = mask_resized.astype(np.uint8)
        color_mask = np.zeros_like(frame)
        color_mask[:, :, 2] = mask_resized
        blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

        # GPS conversion and highlight flooded cell
        flooded_info = self.draw_grid_and_compute_gps(blended, mask_resized // 255, self.current_gps)

        t3 = time.time()

        # LATENCY CALCULATION
        image_processing_ms = (t1 - t0) * 1000
        ai_inference_ms = (t2 - t1) * 1000
        gps_conversion_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        # Update rolling buffers
        self.latency_buffers['image'].append(image_processing_ms)
        self.latency_buffers['ai'].append(ai_inference_ms)
        self.latency_buffers['gps'].append(gps_conversion_ms)
        self.latency_buffers['total'].append(total_ms)

        # FPS calculation
        current_fps = 1.0 / dt if dt > 0 else 0
        self.fps_buffer.append(current_fps)
        avg_fps = np.mean(self.fps_buffer)

        # Overlay latency, FPS, and flooded GPS
        overlay_texts = [
            f"Frame {frame_num}",
            f"Image Proc: {image_processing_ms:.1f} ms",
            f"AI Inference: {ai_inference_ms:.1f} ms",
            f"GPS Conv: {gps_conversion_ms:.1f} ms",
            f"Total: {total_ms:.1f} ms",
            f"Rolling Avg FPS: {avg_fps:.1f}"
        ]
        if flooded_info:
            center, gps = flooded_info
            overlay_texts.append(f"Flooded Cell GPS: ({gps[0]:.6f}, {gps[1]:.6f})")
            # Draw circle around the most flooded cell
            cv2.circle(blended, center, 15, (0, 0, 255), 3)

        for idx, text in enumerate(overlay_texts):
            cv2.putText(blended, text, (10, 20 + idx*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show result
        cv2.imshow("DeepLabV3 Flood Segmentation", blended)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        # Write CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([frame_num, image_processing_ms, ai_inference_ms, gps_conversion_ms, total_ms])

    def draw_grid_and_compute_gps(self, image, mask, current_gps, grid_size=4):
        height, width = mask.shape
        cell_h, cell_w = height // grid_size, width // grid_size
        max_ratio = -1
        max_cell_center = (0, 0)

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_mask = mask[y1:y2, x1:x2]
                flood_ratio = np.mean(cell_mask)

                # Color code grid
                if flood_ratio > 0.5:
                    color = (0, 0, 255)
                elif flood_ratio > 0.2:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                if flood_ratio > max_ratio:
                    max_ratio = flood_ratio
                    max_cell_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Compute GPS for most flooded cell
        if max_ratio > 0.3:
            center_x, center_y = max_cell_center
            pixel = np.array([center_x, center_y, 1])
            ray_cam = self.K_inv @ pixel
            ray_cam /= np.linalg.norm(ray_cam)
            ray_world = self.R_pitch @ ray_cam
            ray_world /= ray_world[2]

            h = self.drone_altitude
            scale = -h / ray_world[2]
            ground_point = scale * ray_world

            dx = ground_point[0]
            dy = -ground_point[2]

            lat, lon = current_gps
            distance = np.sqrt(dx**2 + dy**2)
            azimuth = np.rad2deg(np.arctan2(dx, dy))
            new_point = self.geod.Direct(lat, lon, azimuth, distance)

            target_lat = new_point['lat2']
            target_lon = new_point['lon2']
            print(f"Flood Detected -> Target GPS: ({target_lat:.6f}, {target_lon:.6f})")

            return max_cell_center, (target_lat, target_lon)
        else:
            return None

    def run(self):
        frame_num = 0
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_num += 1
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            self.process_frame(frame, frame_num, dt)

        self.cap.release()


if __name__ == '__main__':
    inference = DeepLabInference()
    inference.run()
