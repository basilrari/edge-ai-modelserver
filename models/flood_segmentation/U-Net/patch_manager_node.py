import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPath, GeoPoseStamped, GeoPose
from cv_bridge import CvBridge
import numpy as np
import cv2
from geographiclib.geodesic import Geodesic
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import torch
from torchvision import transforms
from PIL import Image as PILImage
from model import UNet


class PatchManagerNode(Node):
    def __init__(self):
        super().__init__('patch_manager_node')

        # Parameters for initial GPS origin
        self.declare_parameter('initial_lat', 28.5450)
        self.declare_parameter('initial_lon', 77.1926)

        # Utils
        self.bridge = CvBridge()
        self.geod = Geodesic.WGS84
        self.current_pose = None

        # Camera intrinsics
        self.fx = 277.19
        self.fy = 277.19
        self.cx = 160.5
        self.cy = 120.5
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(self.K)

        # Camera pitch rotation
        theta = np.deg2rad(45)
        self.R_pitch = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # QoS for pose subscription
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos_profile)

        # Publisher (updated to match navigation node)
        self.gps_patch_pub = self.create_publisher(GeoPath, '/flood_patch/waypoints', 10)

        # Load segmentation model
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load('unet_model.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.get_logger().info("[INIT] PatchManagerNode initialized with real-time inference.")

    def pose_callback(self, msg):
        self.current_pose = msg

    def image_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("[POSE] Waiting for drone pose...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"[IMG] Image conversion error: {str(e)}")
            return

        pil_img = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

        mask_resized = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0])) * 255
        self.process_mask(mask_resized.astype(np.uint8))

    def process_mask(self, mask):
        bin_mask = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        patches = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) >= 100]
        patches.sort(reverse=True, key=lambda x: x[0])

        if not patches:
            self.get_logger().info("[DETECT] No valid flood patches detected.")
            return

        geo_path = GeoPath()
        geo_path.header.frame_id = "map"
        geo_path.header.stamp = self.get_clock().now().to_msg()

        drone_z = self.current_pose.pose.position.z
        drone_lat = self.get_parameter('initial_lat').get_parameter_value().double_value
        drone_lon = self.get_parameter('initial_lon').get_parameter_value().double_value

        vis_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for area, cnt in patches:
            cv2.drawContours(vis_image, [cnt], -1, (0, 255, 0), 2)

            for pt in cnt:
                px, py = pt[0]
                pixel = np.array([px, py, 1])
                ray_cam = self.K_inv @ pixel
                ray_cam /= np.linalg.norm(ray_cam)
                ray_world = self.R_pitch @ ray_cam
                ray_world /= ray_world[2]

                scale = -drone_z / ray_world[2]
                ground_point = scale * ray_world

                dx = ground_point[0]
                dy = -ground_point[2]

                distance = np.sqrt(dx**2 + dy**2)
                azimuth = np.rad2deg(np.arctan2(dx, dy))

                result = self.geod.Direct(drone_lat, drone_lon, azimuth, distance)

                geo_pose = GeoPose()
                geo_pose.position.latitude = result['lat2']
                geo_pose.position.longitude = result['lon2']
                geo_pose.position.altitude = 0.0

                geo_pose_stamped = GeoPoseStamped()
                geo_pose_stamped.header.frame_id = "map"
                geo_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                geo_pose_stamped.pose = geo_pose

                geo_path.poses.append(geo_pose_stamped)

                cv2.circle(vis_image, (px, py), 2, (0, 0, 255), -1)

        # Publish to match navigation node topic
        self.gps_patch_pub.publish(geo_path)
        self.get_logger().info(f"[PUBLISH] Sent {len(geo_path.poses)} GPS points from {len(patches)} patch(es) to /flood_patch/waypoints.")

        cv2.imshow("Flood Patch Detection", vis_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PatchManagerNode()
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
