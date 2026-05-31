import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geographic_msgs.msg import GeoPoseStamped
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64MultiArray
import numpy as np
import cvxpy as cp

class MPCPathPlanner(Node):
    def __init__(self):
        super().__init__('mpc_path_planner_node')
        self.get_logger().info("MPC Path Planner Node Started.")

        # Use BEST_EFFORT reliability specifically for GPS subscription to fix QoS warning
        gps_qos = QoSProfile(depth=10)
        gps_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # For other topics, keep default (RELIABLE)
        default_qos = QoSProfile(depth=10)

        self.current_gps = None
        self.target_gps = None
        self.flood_zones = []  # List of (lat, lon) tuples

        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, gps_qos)
        self.create_subscription(GeoPoseStamped, '/next_gps_waypoint', self.target_callback, default_qos)
        self.create_subscription(Float64MultiArray, '/flood_mask_coords', self.flood_mask_callback, default_qos)

        self.publisher = self.create_publisher(GeoPoseStamped, '/mpc_optimized_waypoints', 10)

        self.timer = self.create_timer(1.0, self.solve_mpc)

    def gps_callback(self, msg):
        self.current_gps = (msg.latitude, msg.longitude)

    def target_callback(self, msg):
        self.target_gps = (msg.pose.position.latitude, msg.pose.position.longitude)

    def flood_mask_callback(self, msg):
        coords = msg.data
        self.flood_zones = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

    def is_near_flood_zone(self, lat, lon, threshold=0.0003):
        for flood_lat, flood_lon in self.flood_zones:
            if abs(lat - flood_lat) < threshold and abs(lon - flood_lon) < threshold:
                return True
        return False

    def solve_mpc(self):
        if not self.current_gps or not self.target_gps:
            return

        N = 5  # horizon
        x = cp.Variable(N)
        y = cp.Variable(N)

        x0, y0 = self.current_gps
        xf, yf = self.target_gps

        cost = cp.sum_squares(x - xf) + cp.sum_squares(y - yf)
        constraints = [x[0] == x0, y[0] == y0]

        for t in range(1, N):
            constraints += [cp.abs(x[t] - x[t-1]) <= 0.0005]
            constraints += [cp.abs(y[t] - y[t-1]) <= 0.0005]

        for i in range(N):
            for fx, fy in self.flood_zones:
                dist_sq = (x[i] - fx)**2 + (y[i] - fy)**2
                constraints += [dist_sq >= 0.0001**2]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve()
            for i in range(1, N):
                msg = GeoPoseStamped()
                msg.header.frame_id = 'map'
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.pose.position.latitude = float(x.value[i])
                msg.pose.position.longitude = float(y.value[i])
                msg.pose.position.altitude = 0.0  # Modify if needed
                msg.pose.orientation.w = 1.0
                self.publisher.publish(msg)
                self.get_logger().info(f"[MPC WAYPOINT] Published lat={msg.pose.position.latitude:.6f}, lon={msg.pose.position.longitude:.6f}")
        except Exception as e:
            self.get_logger().error(f"MPC solve failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MPCPathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
