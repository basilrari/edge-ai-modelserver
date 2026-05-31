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

        # Extract flood boundaries and publish GPS points
        self.extract_flood_boundaries_and_publish(blended, mask_resized // 255, self.current_gps)

        cv2.imshow("U-Net Flood Segmentation", blended)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def extract_flood_boundaries_and_publish(self, image, mask, current_gps):
        # Find contours of the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lat, lon = current_gps
        h = self.drone_altitude

        boundary_points_gps = []

        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 100:
                continue

            # Draw contour on image for visualization
            cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)

            # Iterate through contour points
            for point in contour:
                px, py = point[0]  # contour point pixel coordinates

                pixel = np.array([px, py, 1])

                # Compute ray in camera coords
                ray_cam = self.K_inv @ pixel
                ray_cam /= np.linalg.norm(ray_cam)

                # Rotate to world coordinates
                ray_world = self.R_pitch @ ray_cam
                ray_world /= ray_world[2]

                # Project to ground plane z=0
                scale = -h / ray_world[2]
                ground_point = scale * ray_world

                dx = ground_point[0]   # East
                dy = -ground_point[2]  # North

                # Calculate GPS of this boundary point
                distance = np.sqrt(dx**2 + dy**2)
                azimuth = np.rad2deg(np.arctan2(dx, dy))  # Bearing from North clockwise

                new_point = self.geod.Direct(lat, lon, azimuth, distance)
                target_lat = new_point['lat2']
                target_lon = new_point['lon2']

                boundary_points_gps.append((target_lat, target_lon))

        self.get_logger().info(f"Extracted {len(boundary_points_gps)} flood boundary GPS points")



        # Publish the first boundary point as next waypoint if available
        if boundary_points_gps:
            gps_msg = GeoPoseStamped()
            gps_msg.header.frame_id = "map"
            gps_msg.header.stamp = self.get_clock().now().to_msg()

            first_lat, first_lon = boundary_points_gps[0]
            gps_msg.pose.position = GeoPoint()
            gps_msg.pose.position.latitude = first_lat
            gps_msg.pose.position.longitude = first_lon
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
