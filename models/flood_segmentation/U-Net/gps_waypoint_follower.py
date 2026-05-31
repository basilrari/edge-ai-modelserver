import rclpy
from rclpy.node import Node
from geographic_msgs.msg import GeoPoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import NavSatFix
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class GPSWaypointFollower(Node):
    def __init__(self):
        super().__init__('gps_waypoint_follower')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.waypoint_pub = self.create_publisher(GeoPoseStamped, '/mavros/setpoint_position/global', 10)

        # Subscriptions
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_cb, qos_profile)
        self.next_wp_sub = self.create_subscription(GeoPoseStamped, '/next_gps_waypoint', self.waypoint_cb, 10)

        # Services
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_mode service...')

        # Variables
        self.current_state = State()
        self.current_gps = None
        self.target_waypoint = None
        self.setpoints_sent = 0

        # Timer for loop at 10Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

    def state_cb(self, msg):
        self.current_state = msg

    def gps_cb(self, msg):
        self.current_gps = msg

    def waypoint_cb(self, msg):
        self.target_waypoint = msg
        self.get_logger().info(
            f"Received new target waypoint: {msg.pose.position.latitude:.6f}, {msg.pose.position.longitude:.6f}"
        )

    def publish_waypoint(self):
        if self.target_waypoint is not None:
            self.waypoint_pub.publish(self.target_waypoint)
            self.setpoints_sent += 1

    def timer_callback(self):
        # Pre-flight warm-up setpoints before OFFBOARD
        if self.current_state.mode != 'OFFBOARD':
            if self.setpoints_sent < 10:
                self.publish_waypoint()
                self.get_logger().info(f"Sending warm-up setpoint {self.setpoints_sent}/10 before OFFBOARD switch")
                return
            else:
                req = SetMode.Request()
                req.custom_mode = 'OFFBOARD'
                future = self.set_mode_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                if future.result() and future.result().mode_sent:
                    self.get_logger().info("OFFBOARD mode enabled")
                else:
                    self.get_logger().error("Failed to set OFFBOARD mode")

        # Arming the drone
        if not self.current_state.armed:
            req = CommandBool.Request()
            req.value = True
            future = self.arming_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().success:
                self.get_logger().info("Drone armed")
            else:
                self.get_logger().error("Failed to arm drone")

        # Continue publishing the latest waypoint
        self.publish_waypoint()

def main(args=None):
    rclpy.init(args=args)
    node = GPSWaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
