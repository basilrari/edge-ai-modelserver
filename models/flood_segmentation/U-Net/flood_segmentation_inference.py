import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from geographic_msgs.msg import GeoPoseStamped, GeoPoint
import torch
from torchvision import transforms
import numpy as np
from PIL import Image as PILImage
from geographiclib.geodesic import Geodesic
from model import UNet
import cv2


class CameraInferenceNode(Node):
    def __init__(self):
        super().__init__('camera_segmentation_node')
        self.bridge = CvBridge()

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )

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

        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.gps_callback,
            qos_profile=gps_qos
        )

        self.local_pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.local_pose_callback,
            qos_profile=pose_qos
        )

        # Publishers
        self.mask_publisher = self.create_publisher(
            Image,
            '/flood_mask/image_raw',
            10
        )

        self.waypoint_pub = self.create_publisher(
            GeoPoseStamped,
            '/next_gps_waypoint',
            10
        )

        # State
        self.current_gps = None
        self.drone_altitude = 3.0  # fallback altitude in meters

        # Load U-Net model
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load('unet_model.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Image transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Camera intrinsics
        self.fx = 277.19
        self.fy = 277.19
        self.cx = 160.5
        self.cy = 120.5

        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(K)

        # GeographicLib geodesic object for geodesic calculations
        self.geod = Geodesic.WGS84

        # Rotation matrix for 45 deg pitch about Y axis (to convert camera to world)
        theta = np.deg2rad(45)
        self.R_pitch = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [             0, 1,             0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    def gps_callback(self, msg):
        self.current_gps = (msg.latitude, msg.longitude)

    def local_pose_callback(self, msg):
        self.drone_altitude = msg.pose.position.z
        self.get_logger().info(f"Current drone altitude: {self.drone_altitude:.2f} m")

    def listener_callback(self, msg):
        if self.current_gps is None:
            self.get_logger().info("Waiting for GPS fix...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {str(e)}")
            return

        pil_img = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

        mask_resized = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0])) * 255
        mask_resized = mask_resized.astype(np.uint8)

        try:
            ros_mask_msg = self.bridge.cv2_to_imgmsg(mask_resized, encoding='mono8')
            ros_mask_msg.header = msg.header
            self.mask_publisher.publish(ros_mask_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish mask: {str(e)}")

        # Overlay mask on original image (red channel)
        color_mask = np.zeros_like(cv_image)
        color_mask[:, :, 2] = mask_resized
        blended = cv2.addWeighted(cv_image, 0.7, color_mask, 0.3, 0)

        self.draw_grid_and_log_gps(blended, mask_resized // 255, self.current_gps)

        cv2.imshow("U-Net Flood Segmentation", blended)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def draw_grid_and_log_gps(self, image, mask, current_gps, grid_size=4):
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

                # Color based on flood ratio
                if flood_ratio > 0.5:
                    color = (0, 0, 255)  # Red
                elif flood_ratio > 0.2:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{flood_ratio:.2f}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if flood_ratio > max_ratio:
                    max_ratio = flood_ratio
                    max_cell_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if max_ratio > 0.3:
            center_x, center_y = max_cell_center
            pixel = np.array([center_x, center_y, 1])

            # Ray from camera through pixel in camera coordinates
            ray_cam = self.K_inv @ pixel
            ray_cam /= np.linalg.norm(ray_cam)

            # Rotate to world coordinates (accounting for pitch)
            ray_world = self.R_pitch @ ray_cam
            ray_world /= ray_world[2]

            # Project ray to ground plane (z=0), given drone altitude h
            h = self.drone_altitude
            scale = -h / ray_world[2]
            ground_point = scale * ray_world

            # Corrected offsets
            dx = ground_point[0]             # East
            dy = -ground_point[2]            # North

            lat, lon = current_gps
            distance = np.sqrt(dx**2 + dy**2)
            azimuth = np.rad2deg(np.arctan2(dx, dy))  # Bearing from North clockwise

            new_point = self.geod.Direct(lat, lon, azimuth, distance)
            target_lat = new_point['lat2']
            target_lon = new_point['lon2']

            self.get_logger().info(f"Flood Detected! Publishing next waypoint at: ({target_lat:.6f}, {target_lon:.6f})")

            gps_msg = GeoPoseStamped()
            gps_msg.header.frame_id = "map"
            gps_msg.header.stamp = self.get_clock().now().to_msg()
            gps_msg.pose.position = GeoPoint()
            gps_msg.pose.position.latitude = target_lat
            gps_msg.pose.position.longitude = target_lon
            gps_msg.pose.position.altitude = 0.0

            self.waypoint_pub.publish(gps_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
