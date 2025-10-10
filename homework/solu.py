import cv2
import numpy as np

# ==============================
# 颜色范围类
# ==============================
class ColorRange:
    def __init__(self):
        self.color_ranges = {}

    def add_color_range(self, color_name, lower_hsv, upper_hsv):
        if color_name not in self.color_ranges:
            self.color_ranges[color_name] = []
        self.color_ranges[color_name].append((np.array(lower_hsv), np.array(upper_hsv)))

    def get_mask(self, hsv_img, color_name):
        """获取某种颜色的掩膜（支持多段HSV范围合并）"""
        mask_total = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.inRange(hsv_img, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)
        return mask_total

# ==============================
# 球检测类
# ==============================
class BallDetector:
    def __init__(self, video_path, color_range: ColorRange, roi_rects=None, threshold=0.5):
        """
        roi_rects: [(y1, y2, x1, x2), ...] 多个ROI矩形，将被合并为一个掩膜
        threshold: 主导颜色比例阈值
        """
        self.video_path = video_path
        self.color_range = color_range
        self.roi_rects = roi_rects or []
        self.threshold = threshold

    def make_roi_mask(self, frame_shape):
        """根据多个矩形ROI合并生成单个掩膜"""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        for (y1, y2, x1, x2) in self.roi_rects:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def process_frame(self, frame):
        """处理单帧图像"""
        # 合并多个ROI为一个掩膜
        roi_mask = self.make_roi_mask(frame.shape)
        frame_roi = cv2.bitwise_and(frame, frame, mask=roi_mask)

        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        result = {}
        total_pixels = cv2.countNonZero(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY))
        if total_pixels == 0:
            return None, 0, {}, frame_roi

        for color_name in self.color_range.color_ranges.keys():
            mask = self.color_range.get_mask(hsv, color_name)
            color_pixels = cv2.countNonZero(mask)
            ratio = color_pixels / total_pixels
            result[color_name] = ratio

        # 主导颜色
        dominant_color = max(result, key=result.get)
        dominant_ratio = result[dominant_color]

        detected = dominant_color if dominant_ratio > self.threshold else None
        return detected, dominant_ratio, result, frame_roi

    def draw_rois(self, frame):
        """绘制ROI区域框"""
        for (y1, y2, x1, x2) in self.roi_rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

# ==============================
# 主程序
# ==============================
def main():
    # 定义颜色HSV范围
    color_range = ColorRange()
    color_range.add_color_range("red", (0, 90, 90), (20, 255, 255))
    color_range.add_color_range("red", (160, 100, 100), (179, 255, 255))
    color_range.add_color_range("blue", (100, 70, 50), (130, 255, 255))
    color_range.add_color_range("purple", (130, 50, 80), (160, 255, 255))

    # 定义两个ROI矩形区域（合并使用）
    roi_rects = [
        (110, 310, 200, 270),  # ROI1
        (110, 200, 270, 380),  # ROI2
    ]

    # 初始化检测器（阈值10%）
    detector = BallDetector(r"..\res\output.avi", color_range, roi_rects=roi_rects, threshold=0.075)
    cam = cv2.VideoCapture(detector.video_path)
    cnt = 0

    # 定义颜色映射（BGR）
    color_map = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "purple": (255, 0, 255),
        "none": (0, 255, 0)  # 绿色表示未检测到球
    }

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cnt += 1

        # 处理帧
        detected_color, ratio, result, roi_frame = detector.process_frame(frame)
        frame_with_rois = detector.draw_rois(frame.copy())

        # 根据检测结果设置文字与颜色
        if detected_color:
            text = f"{cnt}: Detected {detected_color.upper()} Ball ({ratio:.2%})"
            text_color = color_map[detected_color]
        else:
            text = f"{cnt}: No Ball ({ratio:.2%})"
            text_color = color_map["none"]

        cv2.putText(frame_with_rois, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        cv2.imshow("Frame", frame_with_rois)
        cv2.imshow("ROI Combined", roi_frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC退出
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
