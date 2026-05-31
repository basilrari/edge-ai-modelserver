import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geographic_msgs.msg import GeoPoseStamped
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import Altitude
from geometry_msgs.msg import PoseStamped
import time
from collections import deque

class AutonomousGPSNavigator(Node):
    def __init__(self, takeoff_altitude_target):
        super().__init__('autonomous_gps_navigation_node')
        self.get_logger().info("Autonomous GPS Navigation Node Started.")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.gps_data = None
        self.amsl = None
        self.local_z = None
        self.takeoff_altitude_target = takeoff_altitude_target
        self.amsl_at_takeoff_start = None
        self.takeoff_mode_active = True
        self.takeoff_completed = False
        self.waypoint_queue = deque()
        self.current_waypoint = None
        self.last_waypoint_time = None
        self.mission_altitude = None
        self.waypoint_count = 1

        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, qos)
        self.create_subscription(Altitude, '/mavros/altitude', self.altitude_callback, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos)
        self.create_subscription(GeoPoseStamped, '/next_gps_waypoint', self.waypoint_callback, qos)

        self.publisher = self.create_publisher(GeoPoseStamped, '/mavros/setpoint_position/global', 10)

        self.wait_for_gps_fix()
        self.get_user_startup_option()

        self.timer = self.create_timer(0.5, self.publish_loop)

    def gps_callback(self, msg):
        self.gps_data = msg

    def altitude_callback(self, msg):
        self.amsl = msg.amsl

    def local_pose_callback(self, msg):
        self.local_z = msg.pose.position.z

    def waypoint_callback(self, msg):
        lat = round(msg.pose.position.latitude, 6)
        lon = round(msg.pose.position.longitude, 6)
        alt = self.mission_altitude if self.mission_altitude else self.amsl
        self.waypoint_queue.append({"lat": lat, "lon": lon, "alt": alt})
        self.get_logger().info(f"[QUEUED] Waypoint {len(self.waypoint_queue)} => lat={lat:.6f}, lon={lon:.6f}")

    def wait_for_gps_fix(self):
        self.get_logger().info("Waiting for GPS fix and altitude...")
        while rclpy.ok() and (self.gps_data is None or self.amsl is None):
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info(f"GPS Fix: lat={self.gps_data.latitude:.6f}, lon={self.gps_data.longitude:.6f}, alt={self.amsl:.2f}")

    def get_user_startup_option(self):
        print("Choose option:")
        print("1. Just take off")
        print("2. Set fixed GPS waypoint")
        choice = input("Enter 1 or 2: ").strip()

        if choice == '1':
            try:
                alt = float(input("Enter target takeoff altitude (in meters): "))
                self.takeoff_altitude_target = alt
                self.mission_altitude = self.amsl + alt
            except ValueError:
                print("Invalid input. Using default altitude = 5.0")
                self.mission_altitude = self.amsl + 5.0
        elif choice == '2':
            lat = float(input("Enter latitude: "))
            lon = float(input("Enter longitude: "))
            alt = float(input("Enter altitude: "))
            self.takeoff_mode_active = False
            self.takeoff_completed = True
            self.mission_altitude = alt
            lat = round(lat, 6)
            lon = round(lon, 6)
            self.waypoint_queue.append({"lat": lat, "lon": lon, "alt": alt})
            self.get_logger().info(f"[SET] Fixed Waypoint => lat={lat:.6f}, lon={lon:.6f}, alt={alt:.2f}")

    def publish_loop(self):
        if self.gps_data is None or self.amsl is None or self.local_z is None:
            return

        msg = GeoPoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        if self.takeoff_mode_active:
            if self.amsl_at_takeoff_start is None:
                self.amsl_at_takeoff_start = self.amsl
                self.mission_altitude = self.amsl + self.takeoff_altitude_target
                self.get_logger().info(f"[TAKEOFF] AMSL fixed: {self.amsl_at_takeoff_start:.2f}, Target Alt: {self.mission_altitude:.2f}")

            msg.pose.position.latitude = round(self.gps_data.latitude, 6)
            msg.pose.position.longitude = round(self.gps_data.longitude, 6)
            msg.pose.position.altitude = self.mission_altitude

            if self.local_z >= self.takeoff_altitude_target:
                self.get_logger().info("[TAKEOFF COMPLETE] Switching to GPS waypoint mode.")
                self.takeoff_mode_active = False
                self.takeoff_completed = True
                self.last_waypoint_time = time.time()

        elif self.waypoint_queue or self.current_waypoint:
            if self.current_waypoint is None:
                self.current_waypoint = self.waypoint_queue.popleft()
                self.last_waypoint_time = time.time()
                self.get_logger().info(f"[TARGETING] Waypoint {self.waypoint_count} => lat={self.current_waypoint['lat']:.6f}, lon={self.current_waypoint['lon']:.6f}")

            msg.pose.position.latitude = self.current_waypoint['lat']
            msg.pose.position.longitude = self.current_waypoint['lon']
            msg.pose.position.altitude = self.mission_altitude

            # Exact GPS match check
            if round(self.gps_data.latitude, 6) == self.current_waypoint['lat'] and \
               round(self.gps_data.longitude, 6) == self.current_waypoint['lon']:
                self.get_logger().info(f"[REACHED] Waypoint {self.waypoint_count} reached (Exact GPS match).")
                self.current_waypoint = None
                self.waypoint_count += 1

                if not self.waypoint_queue:
                    self.get_logger().warn("[MISSION COMPLETE] No more waypoints. Returning to launch.")
                    self.trigger_rtl()
                    return

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)

    def trigger_rtl(self):
        # Optional: publish to /mavros/mission/command or call a service
        self.get_logger().info("[RTL] Triggering Return to Launch by holding position.")
        # Holding position by sending current GPS location as setpoint (can be replaced with actual RTL command)
        msg = GeoPoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.latitude = round(self.gps_data.latitude, 6)
        msg.pose.position.longitude = round(self.gps_data.longitude, 6)
        msg.pose.position.altitude = self.mission_altitude
        msg.pose.orientation.w = 1.0
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = AutonomousGPSNavigator(takeoff_altitude_target=5.0)
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("Shutting down node...")

    if node is not None:
        node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
