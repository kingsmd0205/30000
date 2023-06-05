import rospy
import baxter_interface
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0.0
        self.last_error = 0.0

    def compute_control_output(self, error, dt):
        self.error_sum += error * dt
        error_diff = (error - self.last_error) / dt
        control_output = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        self.last_error = error
        return control_output

def generate_wiping_trajectory(start_point, end_point, num_waypoints):
    """
    生成擦拭运动轨迹

    参数:
    start_point: 起始点的三维坐标 (x, y, z)
    end_point: 终点的三维坐标 (x, y, z)
    num_waypoints: 轨迹中的中间点数量

    返回:
    trajectory: 生成的擦拭运动轨迹，包含起始点、中间点和终点的三维坐标数组
    """

    # 生成等间距的中间点
    waypoints = np.linspace(start_point, end_point, num_waypoints)

    # 将起始点、中间点和终点组合成完整的轨迹
    trajectory = np.vstack((start_point, waypoints, end_point))

    return trajectory

def move_to_joint_positions(limb, joint_angles, timeout=15.0):
    limb.move_to_joint_positions(joint_angles, timeout=timeout)

def execute_trajectory(limb, trajectory):
    pid_controller = PIDController(kp=1.0, ki=0.0, kd=0.0)  # 设置适当的PID参数

    for point in trajectory:
        joint_angles = limb.ik_request(point)

        # 控制器计算控制输出
        current_position = np.array(limb.endpoint_pose()['position'])
        error = point - current_position
        control_output = pid_controller.compute_control_output(error, dt=0.01)  # 设置适当的时间步长

        # 将控制输出添加到关节角度上
        joint_angles = {joint: angle + control_output for joint, angle in joint_angles.items()}

        move_to_joint_positions(limb, joint_angles)

def main():
    rospy.init_node('wiping_robot')

    # 初始化Baxter机器人
    limb = baxter_interface.Limb('right')
    limb.set_joint_position_speed(0.2)

    # 定义起始点、终点和中间点数量
    start_point = np.array([0.1, 0.2, 0.3])
    end_point = np.array([0.4, 0.5, 0.6])
    num_waypoints = 5

    # 生成擦拭运动轨迹
    trajectory = generate_wiping_trajectory(start_point, end_point, num_waypoints)

    # 执行擦拭运动轨迹
    execute_trajectory(limb, trajectory)

if __name__ == '__main__':
    main()

