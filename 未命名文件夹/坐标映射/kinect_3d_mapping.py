import cv2
import numpy as np
from pykinect2 import PyKinectRuntime
from subprocess import check_output

def map_2d_to_3d(kinect, depth_image, depth_scale=1.0):
    points_2d = np.nonzero(depth_image)
    points_2d = np.array(points_2d).T

    points_3d = []
    for point_2d in points_2d:
        depth = depth_image[point_2d[0], point_2d[1]]
        x = point_2d[1]
        y = point_2d[0]
        x, y, z = kinect._mapper.MapDepthPointToCameraSpace(x, y, depth * depth_scale)
        points_3d.append([x, y, z])

    return np.array(points_3d)

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectRuntime.FrameSourceTypes_Depth)
while True:
    if kinect.has_new_depth_frame():
        depth_frame = kinect.get_last_depth_frame()
        depth_image = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))

        # 将 depth_image 保存为临时文件
        depth_image_path = 'depth_image.png'
        cv2.imwrite(depth_image_path, depth_image)

        # 运行 copyUNET-live10.py 脚本获取分割图像
        command = ['python', 'copyUNET-live10.py', '--input', depth_image_path, '--output', 'segmented_image.png']
        check_output(command)

        # 读取分割图像
        segmented_image = cv2.imread('segmented_image.png', cv2.IMREAD_GRAYSCALE)

        points_3d = map_2d_to_3d(kinect, depth_image)

        # 在原始深度图像上绘制分割结果
        segmented_image_color = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
        for point_3d in points_3d:
            x, y, z = point_3d
            if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                x = int(x)
                y = int(y)
                cv2.circle(segmented_image_color, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow('Segmentation', segmented_image_color)
        cv2.waitKey(1)

