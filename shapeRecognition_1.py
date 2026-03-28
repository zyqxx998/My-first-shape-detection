# 须知：测试图片为豆包生成，数据有待考究！！！
import cv2
import numpy as np

class ShapeDetector:
    def __init__(self):
        # 初始化检测器
        pass

    def detect(self, contour):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        # 过滤极小噪点 (1000为面积像素阈值，可根据相机距离微调)
        if perimeter == 0 or area < 1000: 
            return "None", 0, 0
        
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # 核心数学特征：圆度
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # 核心数学特征：外接矩形长宽比
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # --- 基于台面物体(300mm)实测投影畸变的判定逻辑 ---

        # 1. 球体判定 (高优先级：二维投影始终为圆)
        if circularity > 0.88:
            return "Ball", circularity, vertices

        # 2. 正方体与圆柱体判定 (基于1:1尺寸的透视畸变分水岭)
        # 官方尺寸皆为 300x300，投影长宽比通常在 0.7~1.3 之间
        if 0.7 <= aspect_ratio <= 1.3:
            # 根据实测视角：六边形投影(方块)圆度较高，胶囊投影(圆柱)圆度较低
            # 0.76 为实测黄金分水岭
            if circularity > 0.76:
                return "Cube", circularity, vertices
            else:
                return "Cylinder", circularity, vertices

        # 3. 兜底逻辑：极度偏视角的圆柱体
        if 0.4 < circularity <= 0.76:
            return "Cylinder", circularity, vertices
            
        return "Unknown", circularity, vertices