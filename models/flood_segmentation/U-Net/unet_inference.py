import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image as PILImage
from geographiclib.geodesic import Geodesic
import time
from model import UNet


class UNetInference:

    def __init__(self):

        # ========================
        # CAMERA (same as DeepLab)
        # ========================
        self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            raise RuntimeError("Camera not opening")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ========================
        # DEVICE
        # ========================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========================
        # MODEL
        # ========================
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load("unet_model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # ========================
        # CAMERA INTRINSICS
        # ========================
        self.fx, self.fy = 277.19, 277.19
        self.cx, self.cy = 160.5, 120.5

        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(K)

        # ========================
        # GPS (dummy like DeepLab)
        # ========================
        self.geod = Geodesic.WGS84
        self.current_gps = (37.7749, -122.4194)
        self.altitude = 3.0

        # rotation
        theta = np.deg2rad(45)
        self.R_pitch = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        self.prev_time = time.time()

        print("UNet DeepLab-style inference initialized")

    # =========================================================
    def process_frame(self, frame):

        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            mask = torch.sigmoid(out).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        centroid, gps_text = self.analyze_grid(mask)

        color = np.zeros_like(frame)
        color[:, :, 2] = mask * 255

        blended = cv2.addWeighted(frame, 0.7, color, 0.3, 0)

        if centroid:
            cv2.circle(blended, centroid, 8, (0, 0, 255), -1)

        if gps_text:
            cv2.putText(blended, gps_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

        # FPS
        fps = 1 / (time.time() - self.prev_time + 1e-6)
        self.prev_time = time.time()

        cv2.putText(blended, f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.imshow("UNet Flood Detection", blended)

    # =========================================================
    def analyze_grid(self, mask, grid_size=4):

        h, w = mask.shape
        ch, cw = h // grid_size, w // grid_size

        best = -1
        best_cell = None

        for i in range(grid_size):
            for j in range(grid_size):

                y1, y2 = i * ch, (i + 1) * ch
                x1, x2 = j * cw, (j + 1) * cw

                cell = mask[y1:y2, x1:x2]
                ratio = np.mean(cell)

                if ratio > best:
                    best = ratio
                    best_cell = (x1, y1, x2, y2)

        if best < 0.3:
            return None, None

        x1, y1, x2, y2 = best_cell

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        pixel = np.array([cx, cy, 1])
        ray = self.K_inv @ pixel
        ray /= np.linalg.norm(ray)

        ray_world = self.R_pitch @ ray
        ray_world /= ray_world[2]

        scale = -self.altitude / ray_world[2]
        ground = scale * ray_world

        dx, dy = ground[0], -ground[2]

        lat, lon = self.current_gps
        dist = np.sqrt(dx**2 + dy**2)
        az = np.rad2deg(np.arctan2(dx, dy))

        new = self.geod.Direct(lat, lon, az, dist)

        gps_text = f"GPS: {new['lat2']:.6f}, {new['lon2']:.6f}"

        print(gps_text)

        return (cx, cy), gps_text

    # =========================================================
    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# =========================================================
if __name__ == "__main__":
    app = UNetInference()
    app.run()
