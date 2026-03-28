import cv2
import numpy as np
from shapeRecognition_1 import ShapeDetector

# 初始化识别引擎
detector = ShapeDetector()

# 读取现场图像 (请替换为机器人的实时抽帧图片)
img_path = 'shapetest2.jpg' 
raw_img = cv2.imread(img_path)

if raw_img is None:
    print(f"致命错误: 无法加载图像 {img_path}")
    exit()

# --- 图像预处理流水线 ---
gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# 实测环境光黄金阈值：80 (切断淡阴影)
_, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

# 形态学手术刀：切断微弱粘连，保持物体独立
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# --- 轮廓提取与识别 ---
display_img = raw_img.copy()
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("\n=== CRAIC 具身智能任务赛 - 视觉测试日志 ===")
print(f"{'识别物体':<10} | {'实测圆度(C)':<12} | {'拟合顶点(V)':<8}")
print("-" * 45)

for c in cnts:
    res, circ, vert = detector.detect(c)
    
    if res == "None": continue

    # 打印核心诊断数据
    print(f"{res:<12} | {circ:<12.3f} | {vert:<8}")

    # 画面可视化绘制
    cv2.drawContours(display_img, [c], -1, (0, 255, 0), 2)
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # 在图像上打标签
        label = f"{res} C:{circ:.2f}"
        cv2.putText(display_img, label, (cX-40, cY), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# --- 结果呈现 ---
cv2.imshow("Binary Mask (Pre-processing View)", thresh)
cv2.imshow("Robot Camera View", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()