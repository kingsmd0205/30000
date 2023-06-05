import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs import point_cloud2 as pc2
import tf.transformations as tr
import numpy as np
import tf.transformations as tr
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics

class BaxterDMP(object):
    def __init__(self, limb):
        self._arm = baxter_interface.limb.Limb(limb)
        self._name = self._arm.joint_names()
        self._kin = baxter_kinematics(limb)
        self._joint_angle = dict()
        self._filename1 = '/home/robot707/ros_ws/src/visual_servoing_pbvs/src/dmpResult1.csv'
        self._filename2 = '/home/robot707/ros_ws/src/visual_servoing_pbvs/src/recording1.csv'
        self._main()

    def set_joint_angles(self, angles):
        joints = dict(zip(self._name, angles))
        self._arm.move_to_joint_positions(joints, timeout=15.0)

    def _trajectory(self, name, euler=True):
        tra = np.loadtxt(name, delimiter=",", dtype=float)
        Nt = len(tra)
        if euler:
            tra_euler = tra[0:Nt, 3:6]
            tra_quar = []
            for i in range(Nt):
                temp = tr.quaternion_from_euler(tra_euler[i][0], tra_euler[i][1], tra_euler[i][2])
                tra_quar.append(temp.tolist())
            tra = np.concatenate((tra[0:Nt, 0:3], tra_quar), axis=1)
            return tra, Nt
        else:
            return tra, Nt

    def _main(self):
        tra, Nt = self._trajectory(self._filename1, euler=True)
        tra = tra.tolist()
        for i in range(Nt):
            position = tra[i][0:3]
            orientation = tra[i][3:7]
            test = self._kin.inverse_kinematics(position, orientation)
            self.set_joint_angles(test)

def process_and_transform(points_3d, limb):
    kinect_frame = 'kinect_frame'
    baxter_base_frame = 'base'
    listener = tf.TransformListener()
    
    # 获取Kinect到Baxter基坐标系的变换矩阵
    kinect_to_baxter = listener.lookup_transform(baxter_base_frame, kinect_frame, rospy.Time(0))
    kinect_to_baxter_matrix = tr.quaternion_matrix([kinect_to_baxter.transform.rotation.x, kinect_to_baxter.transform.rotation.y, kinect_to_baxter.transform.rotation.z, kinect_to_baxter.transform.rotation.w])
    kinect_to_baxter_matrix[0][3] = kinect_to_baxter.transform.translation.x
    kinect_to_baxter_matrix[1][3] = kinect_to_baxter.transform.translation.y
    kinect_to_baxter_matrix[2][3] = kinect_to_baxter.transform.translation.z

    arm = baxter_interface.limb.Limb(limb)
    kinematics = baxter_kinematics(limb)
    joint_angles = []
    
    for point in points_3d:
        # 将点从Kinect坐标系转换到Baxter基坐标系下的笛卡尔坐标系
        kinect_point = np.array([[point[0], point[1], point[2], 1]])
        baxter_point = np.dot(kinect_to_baxter_matrix, kinect_point.T)

        # 将点从笛卡尔坐标系转换到关节角
        position_ik = [baxter_point[0][0], baxter_point[1][0], baxter_point[2][0]]
        quaternion_ik = [1, 0, 0, 0]
        angles = kinematics.inverse_kinematics(position_ik, quaternion_ik)

        joint_angles.append(angles)
    
    # 调用Baxter DMP类进行轨迹规划和执行
    bdmp = BaxterDMP(limb)
    joint_angles_with_start_and_end = [joint_angles[0]] + joint_angles + [joint_angles[-1]]
    for angles in joint_angles_with_start_and_end:
        bdmp.set_joint_angles(angles)

def point_cloud_callback(data):
    point_cloud = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
    points_3d = []
    for point in point_cloud:
        x = point[0]
        y = point[1]
        z = point[2]
        points_3d.append((x, y, z))
    process_and_transform(points_3d, 'right')

# 初始化ROS节点
rospy.init_node('kinect_to_baxter')

# 订阅Kinect的点云数据
rospy.Subscriber('/kinect/point_cloud', PointCloud2, point_cloud_callback)

# 循环等待ROS消息
rospy.spin()

