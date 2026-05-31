import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.models.segmentation as models
from PIL import Image as PILImage
from geographiclib.geodesic import Geodesic
import time


class DeepLabInference:
    def __init__(self, save_frames=False):

        # ========================
        # CAMERA
        # ========================
        self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            raise RuntimeError("Camera not opening. Try index 0 / 1 / 2")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ========================
        # DEVICE
        # ========================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========================
        # MODEL
        # ========================
        self.model = models.deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # ========================
        # CAMERA INTRINSICS
        # ========================
        self.fx, self.fy = 277.19, 277.19
        self.cx, self.cy = 160.5, 120.5

        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]])
        self.K_inv = np.linalg.inv(K)

        # ========================
        # GPS REFERENCE
        # ========================
        self.geod = Geodesic.WGS84
        self.current_gps = (37.7749, -122.4194)
        self.drone_altitude = 3.0

        # ========================
        # OTHER
        # ========================
        self.prev_time = time.time()
        self.save_frames = save_frames
        self.frame_count = 0

        print("System Initialized Successfully!")

    # =========================================================
    # MAIN FRAME PROCESSING
    # =========================================================
    def process_frame(self, frame):

        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)['out']
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        mask = cv2.resize(pred.astype(np.uint8),
                          (frame.shape[1], frame.shape[0]))

        # ========================
        # GRID + CENTROID + GPS
        # ========================
        result = self.analyze_grid(mask)
        centroid = result["centroid"]
        gps_text = result["gps_text"]

        # ========================
        # OVERLAY MASK
        # ========================
        color_mask = np.zeros_like(frame)
        color_mask[:, :, 2] = mask * 255
        blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

        # ========================
        # DRAW CENTROID (RED DOT)
        # ========================
        if centroid is not None:
            cv2.circle(blended, centroid, 8, (0, 0, 255), -1)

        # ========================
        # DISPLAY GPS TEXT
        # ========================
        if gps_text is not None:
            cv2.putText(blended, gps_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

        # ========================
        # FPS
        # ========================
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time

        cv2.putText(blended, f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        # ========================
        # SHOW
        # ========================
        cv2.imshow("DeepLab Flood Detection", blended)

        # optional save
        if self.save_frames and self.frame_count % 30 == 0:
            cv2.imwrite(f"frame_{self.frame_count}.jpg", blended)

        self.frame_count += 1

    # =========================================================
    # 4x4 GRID + CENTROID + GPS
    # =========================================================
    def analyze_grid(self, mask, grid_size=4):

        h, w = mask.shape
        cell_h, cell_w = h // grid_size, w // grid_size

        max_ratio = -1
        best_cell = None

        for i in range(grid_size):
            for j in range(grid_size):

                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w

                cell = mask[y1:y2, x1:x2]
                ratio = np.mean(cell)

                if ratio > max_ratio:
                    max_ratio = ratio
                    best_cell = (x1, y1, x2, y2)

        if max_ratio < 0.3:
            return {"centroid": None, "gps_text": None}

        x1, y1, x2, y2 = best_cell

        # centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        centroid = (cx, cy)

        # ========================
        # GPS ESTIMATION
        # ========================
        pixel = np.array([cx, cy, 1])
        ray_cam = self.K_inv @ pixel
        ray_cam /= np.linalg.norm(ray_cam)

        theta = np.deg2rad(45)

        R_pitch = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        ray_world = R_pitch @ ray_cam
        ray_world /= ray_world[2]

        scale = -self.drone_altitude / ray_world[2]
        ground_point = scale * ray_world

        dx, dy = ground_point[0], -ground_point[2]

        lat, lon = self.current_gps
        distance = np.sqrt(dx**2 + dy**2)
        azimuth = np.rad2deg(np.arctan2(dx, dy))

        new_point = self.geod.Direct(lat, lon, azimuth, distance)

        gps_text = f"GPS: {new_point['lat2']:.6f}, {new_point['lon2']:.6f}"

        print(gps_text)

        return {
            "centroid": centroid,
            "gps_text": gps_text
        }

    # =========================================================
    # RUN LOOP
    # =========================================================
    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                print("Frame grab failed")
                break

            self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    inference = DeepLabInference(save_frames=True)
    inference.run()
