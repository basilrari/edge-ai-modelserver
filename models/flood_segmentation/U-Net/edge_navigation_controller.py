import rclpy
from rclpy.node import Node
from geographic_msgs.msg import GeoPath
from mavros_msgs.msg import GlobalPositionTarget
from mavros_msgs.srv import CommandLong
from enum import Enum
import time

# Configurable constants
TAKEOFF_ALTITUDE = 10.0  # meters
WAYPOINT_HOLD_TIME = 5.0  # seconds

class FlightState(Enum):
    TAKEOFF = 1
    NAVIGATE = 2
    RETURN_HOME = 3
    IDLE = 4

class EdgeNavigator(Node):
    def __init__(self):
        super().__init__('edge_navigation_controller')

        # Subscribers
        self.create_subscription(GeoPath, '/flood_patch/waypoints', self.waypoints_callback, 10)

        # Publishers
        self.gps_pub = self.create_publisher(GlobalPositionTarget, '/mavros/setpoint_position/global', 10)

        # Services
        self.rtl_client = self.create_client(CommandLong, '/mavros/cmd/command')

        # Internal state
        self.flight_state = FlightState.IDLE
        self.current_path = []
        self.current_wp_index = 0
        self.last_wp_time = None
        self.reached_takeoff_altitude = False

        # Main loop
        self.timer = self.create_timer(1.0, self.navigation_loop)

        self.get_logger().info("[INIT] Edge Navigation Controller initialized.")

    def waypoints_callback(self, msg):
        if not msg.poses:
            self.get_logger().warn("[WAYPOINT] Received empty waypoint list.")
            return

        self.get_logger().info(f"[WAYPOINT] Received {len(msg.poses)} GPS waypoints for patch survey.")

        self.current_path = msg.poses
        self.current_wp_index = 0
        self.last_wp_time = None
        self.flight_state = FlightState.TAKEOFF
        self.reached_takeoff_altitude = False

    def navigation_loop(self):
        if not self.current_path:
            return  # No mission loaded yet

        if self.flight_state == FlightState.TAKEOFF:
            self.takeoff()
        elif self.flight_state == FlightState.NAVIGATE:
            self.navigate_patch()
        elif self.flight_state == FlightState.RETURN_HOME:
            self.trigger_rtl()
        elif self.flight_state == FlightState.IDLE:
            pass  # Waiting for new mission

    def takeoff(self):
        if not self.reached_takeoff_altitude:
            first_wp = self.current_path[0].pose.position
            self.publish_gps_target(first_wp.latitude, first_wp.longitude, TAKEOFF_ALTITUDE)
            self.get_logger().info(f"[TAKEOFF] Climbing to {TAKEOFF_ALTITUDE}m over first waypoint...")
            # In real scenario, check altitude via /mavros/local_position/pose
            self.reached_takeoff_altitude = True
            return

        self.get_logger().info("[TAKEOFF] Altitude reached. Starting patch navigation.")
        self.flight_state = FlightState.NAVIGATE

    def navigate_patch(self):
        if self.current_wp_index >= len(self.current_path):
            self.get_logger().info("[NAV] Completed all waypoints. Initiating RTL.")
            self.flight_state = FlightState.RETURN_HOME
            return

        wp = self.current_path[self.current_wp_index].pose.position
        self.publish_gps_target(wp.latitude, wp.longitude, TAKEOFF_ALTITUDE)

        if self.last_wp_time is None:
            self.last_wp_time = time.time()
            self.get_logger().info(f"[NAV] Moving to waypoint {self.current_wp_index + 1}/{len(self.current_path)}")

        # Simulated "arrival" after hold time
        if time.time() - self.last_wp_time >= WAYPOINT_HOLD_TIME:
            self.current_wp_index += 1
            self.last_wp_time = None

    def publish_gps_target(self, lat, lon, alt):
        msg = GlobalPositionTarget()
        msg.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        # type_mask: ignore velocity, accel, yaw, yaw_rate → only use position
        msg.type_mask = (GlobalPositionTarget.IGNORE_VX |
                         GlobalPositionTarget.IGNORE_VY |
                         GlobalPositionTarget.IGNORE_VZ |
                         GlobalPositionTarget.IGNORE_AFX |
                         GlobalPositionTarget.IGNORE_AFY |
                         GlobalPositionTarget.IGNORE_AFZ |
                         GlobalPositionTarget.FORCE |
                         GlobalPositionTarget.IGNORE_YAW |
                         GlobalPositionTarget.IGNORE_YAW_RATE)
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        self.gps_pub.publish(msg)

    def trigger_rtl(self):
        if not self.rtl_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("[RTL] Service /mavros/cmd/command not available!")
            return

        req = CommandLong.Request()
        req.command = 20  # MAV_CMD_NAV_RETURN_TO_LAUNCH
        req.confirmation = 0
        req.param1 = 0.0

        self.get_logger().info("[RTL] Sending Return to Launch command...")

        future = self.rtl_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().success:
            self.get_logger().info("[RTL] RTL command sent successfully.")
        else:
            self.get_logger().error("[RTL] Failed to send RTL command.")

        self.flight_state = FlightState.IDLE
        self.current_path = []  # Clear mission

def main(args=None):
    rclpy.init(args=args)
    node = EdgeNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
