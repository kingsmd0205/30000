
import cv2
import numpy as np
import tensorflow as tf

# 加载神经网络模型
def load_model(model_path):
    # 你的模型加载代码
    model = tf.keras.models.load_model(model_path)
    return model

# 对输入图像进行分割
def segment_image(image, model):
    # 图像预处理，例如缩放、归一化等
    preprocessed_image = preprocess_image(image)
    
    # 使用模型进行预测
    segmentation = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # 处理分割结果，例如阈值化、后处理等
    processed_segmentation = process_segmentation(segmentation)
    
    return processed_segmentation

# 图像预处理
def preprocess_image(image):
    # 图像预处理操作，例如缩放、归一化等
    preprocessed_image = image / 255.0  # 示例：归一化到 [0, 1]
    return preprocessed_image

# 处理分割结果
def process_segmentation(segmentation):
    # 分割结果处理操作，例如阈值化、后处理等
    processed_segmentation = (segmentation > 0.5).astype(np.uint8)  # 示例：二值化
    return processed_segmentation

# 主函数
def main():
    # 设置相关参数和路径
    model_path = '/home/ros/robot/实时神经网络分割/robot.h5'  # 模型文件路径
    
    # 加载模型
    model = load_model(model_path)
    
    # 打开摄像头（或读取图像）
    cap = cv2.VideoCapture(0)  # 示例：打开摄像头
    
    while True:
        # 读取图像帧
        ret, frame = cap.read()
        
        # 进行分割
        segmentation = segment_image(frame, model)
        
        # 可选择性地在原始图像上绘制分割结果
        result = cv2.bitwise_and(frame, frame, mask=segmentation)
        
        # 显示结果
        cv2.imshow('Segmentation Result', result)
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

