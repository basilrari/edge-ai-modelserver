import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from geographic_msgs.msg import GeoPoseStamped, GeoPoint
import torch
from torchvision import transforms
import numpy as np
from PIL import Image as PILImage
from geographiclib.geodesic import Geodesic
from model import UNet
import cv2


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0.0:
        return np.eye(3)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),         2.0*(xz + wy)],
        [    2.0*(xy + wz),   1.0 - 2.0*(xx + zz),       2.0*(yz - wx)],
        [    2.0*(xz - wy),       2.0*(yz + wx),     1.0 - 2.0*(xx + yy)]
    ], dtype=np.float64)
    return R


def rotmat_to_euler_zyx(R: np.ndarray):
    # ZYX (yaw, pitch, roll) extraction; returns (roll, pitch, yaw) for logging
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


class FloodCellGpsLocalizationNode(Node):
    """
    Segments flood, picks the most flooded grid cell, projects its centroid to the ground
    using camera intrinsics + full UAV attitude, then publishes a GPS waypoint at that point.
    """

    def __init__(self):
        super().__init__('flood_cell_gps_localization_node')
        self.bridge = CvBridge()

        # --- Subscribers ---
        self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)

        gps_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(NavSatFix, '/mavros/global_position/global',
                                 self.gps_callback, qos_profile=gps_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.local_pose_callback, qos_profile=pose_qos)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg',
                                 self.heading_callback, qos_profile=gps_qos)

        # --- Publishers ---
        self.mask_publisher = self.create_publisher(Image, '/flood_mask/image_raw', 10)
        self.waypoint_pub = self.create_publisher(GeoPoseStamped, '/next_gps_waypoint', 10)

        # --- State ---
        self.current_gps = None
        self.drone_altitude = 3.0          # ENU meters
        self.heading_deg = 0.0             # compass heading for logging (0=N, CW+)
        self.R_body_to_enu = np.eye(3)     # from quaternion
        self.euler_rpy_rad = (0.0, 0.0, 0.0)

        # --- Camera intrinsics (update to your calibration) ---
        self.fx = 277.19
        self.fy = 277.19
        self.cx = 160.5
        self.cy = 120.5

        # WGS-84 geodesic
        self.geod = Geodesic.WGS84

        # --- Camera mount: 45° pitch down from +X (body frame) ---
        # Camera frame we use: x_cam = forward, y_cam = right, z_cam = down.
        theta = np.deg2rad(45.0)
        self.R_cam_to_body = np.array([
            [ np.cos(theta), 0,  np.sin(theta)],  # body x (forward)
            [             0, 1,              0],  # body y (right)
            [-np.sin(theta), 0,  np.cos(theta)]   # body z (up)
        ], dtype=np.float64)

        # --- Segmentation model ---
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load('unet_model.pth', map_location=torch.device('cpu')))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # --- Tunables ---
        self.min_alt_for_proj = 0.3  # meters; avoid crazy intersections if almost on ground
        self.min_confidence_ratio = 0.3  # min flooded ratio to trust a cell

    # --- Callbacks ---
    def heading_callback(self, msg: Float64):
        self.heading_deg = float(msg.data)  # diagnostics only

    def gps_callback(self, msg: NavSatFix):
        self.current_gps = (float(msg.latitude), float(msg.longitude))

    def local_pose_callback(self, msg: PoseStamped):
        self.drone_altitude = float(msg.pose.position.z)  # ENU up
        q = msg.pose.orientation
        self.R_body_to_enu = quat_to_rotmat(q.x, q.y, q.z, q.w)
        self.euler_rpy_rad = rotmat_to_euler_zyx(self.R_body_to_enu)

    def listener_callback(self, msg: Image):
        if self.current_gps is None:
            self.get_logger().info("Waiting for GPS fix...")
            return

        # Guard: need some altitude to intersect ground safely
        if self.drone_altitude < self.min_alt_for_proj:
            self.get_logger().warn(f"Altitude {self.drone_altitude:.2f} m too low for stable projection.")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {str(e)}")
            return

        # -------- Segmentation --------
        pil_img = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            bin_mask = (mask > 0.5).astype(np.uint8)

        # Resize mask back to original resolution
        mask_resized = cv2.resize(
            bin_mask, (cv_image.shape[1], cv_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8) * 255

        # Publish mask for visualization
        try:
            ros_mask_msg = self.bridge.cv2_to_imgmsg(mask_resized, encoding='mono8')
            ros_mask_msg.header = msg.header
            self.mask_publisher.publish(ros_mask_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish mask: {str(e)}")

        # Overlay
        color_mask = np.zeros_like(cv_image)
        color_mask[:, :, 2] = mask_resized
        blended = cv2.addWeighted(cv_image, 0.7, color_mask, 0.3, 0)

        # Compute waypoint from grid centroid
        self.process_grid_and_publish_waypoint(
            blended,
            (mask_resized // 255).astype(np.uint8),
            self.current_gps
        )

        # Optional local viz (safe in GUI env)
        try:
            cv2.imshow("Flood Segmentation & Cell Centroid Localization", blended)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        except Exception:
            pass

    # --- Grid processing ---
    def process_grid_and_publish_waypoint(self, image, bin_mask, current_gps, grid_size=4):
        height, width = bin_mask.shape
        cell_h, cell_w = height // grid_size, width // grid_size

        max_ratio = -1.0
        best_bbox = (0, 0, 0, 0)

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = bin_mask[y1:y2, x1:x2]
                flood_ratio = float(np.mean(cell))

                # Colorize grid cell based on flood ratio
                if flood_ratio > 0.5:
                    color = (0, 0, 255)    # Red (heavily flooded)
                elif flood_ratio > 0.2:
                    color = (0, 255, 255)  # Yellow (moderate)
                else:
                    color = (0, 255, 0)    # Green (low)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{flood_ratio:.2f}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if flood_ratio > max_ratio:
                    max_ratio = flood_ratio
                    best_bbox = (x1, y1, x2, y2)

        if max_ratio <= self.min_confidence_ratio:
            # Not enough flooded area to trust
            return

        x1, y1, x2, y2 = best_bbox
        cx_pix = (x1 + x2) // 2
        cy_pix = (y1 + y2) // 2

        # Visualize centroid
        cv2.circle(image, (cx_pix, cy_pix), 5, (255, 0, 0), -1)
        cv2.putText(image, "centroid", (cx_pix + 6, cy_pix - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Compute ENU offset via full attitude-aware ground intersection
        ok, dx_east, dy_north, body_forward, body_right = self.pixel_to_ground_offset_ENU(cx_pix, cy_pix)
        if not ok:
            self.get_logger().warn("Ray does not intersect ground (check attitude/altitude).")
            return

        # ENU offset -> geodesic target
        distance_m = float(np.hypot(dx_east, dy_north))
        azimuth = (np.degrees(np.arctan2(dx_east, dy_north)) + 360.0) % 360.0

        lat, lon = current_gps
        new_point = self.geod.Direct(lat, lon, azimuth, distance_m)
        target_lat = new_point['lat2']
        target_lon = new_point['lon2']

        # Diagnostics
        roll, pitch, yaw = self.euler_rpy_rad
        self.get_logger().info(
            f"Flood centroid -> ENU dx={dx_east:.2f}, dy={dy_north:.2f}, dist={distance_m:.2f}, "
            f"az={azimuth:.1f} => ({target_lat:.6f}, {target_lon:.6f}) | "
            f"compass={self.heading_deg:.1f} deg | "
            f"body_fwd={body_forward:.2f}, body_right={body_right:.2f} | "
            f"RPY(deg)=({np.degrees(roll):.1f}, {np.degrees(pitch):.1f}, {np.degrees(yaw):.1f})"
        )

        gps_msg = GeoPoseStamped()
        gps_msg.header.frame_id = "map"
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.pose.position = GeoPoint()
        gps_msg.pose.position.latitude = target_lat
        gps_msg.pose.position.longitude = target_lon
        gps_msg.pose.position.altitude = 0.0
        self.waypoint_pub.publish(gps_msg)

    # --- Projection (attitude-compensated) ---
    def pixel_to_ground_offset_ENU(self, u: int, v: int):
        """
        Camera model:
          - x_cam = forward, y_cam = right, z_cam = down
          - image u increases to the right, v increases downward
        Steps:
          1) Pixel -> normalized ray in CAMERA: [1, -(u-cx)/fx, -(v-cy)/fy]
             (note the minus on y_cam: left/right fix)
          2) CAMERA -> BODY via gimbal pitch (R_cam_to_body)
          3) BODY -> ENU via quaternion (R_body_to_enu)
          4) Intersect ENU ray with ground plane z=0 using altitude h
        Returns (ok, dx_east, dy_north, body_forward, body_right)
        """
        # Build normalized ray in camera frame
        x_cam = 1.0
        y_cam = -(u - self.cx) / self.fx   # LEFT/RIGHT FIX: minus sign
        z_cam = -(v - self.cy) / self.fy   # v down => z_cam negative when v > cy
        ray_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
        nrm = np.linalg.norm(ray_cam)
        if nrm < 1e-9:
            return (False, 0.0, 0.0, 0.0, 0.0)
        ray_cam /= nrm

        # CAMERA -> BODY
        ray_body = self.R_cam_to_body @ ray_cam  # [forward, right, up] in body

        # BODY -> ENU
        ray_enu = self.R_body_to_enu @ ray_body  # [east, north, up]

        # Must point downward in ENU to hit ground z=0
        if ray_enu[2] >= -1e-6:
            return (False, 0.0, 0.0, 0.0, 0.0)

        # Intersect with ground
        h = float(self.drone_altitude)
        t = -h / ray_enu[2]  # ray_enu[2] < 0
        hit_enu = t * ray_enu

        dx_east = float(hit_enu[0])
        dy_north = float(hit_enu[1])

        # Diagnostics in body axes (same t)
        hit_body = t * ray_body
        body_forward = float(hit_body[0])
        body_right = float(hit_body[1])

        return (True, dx_east, dy_north, body_forward, body_right)


def main(args=None):
    rclpy.init(args=args)
    node = FloodCellGpsLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
            node.destroy_node()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            rclpy.shutdown()


if __name__ == '__main__':
    main()
