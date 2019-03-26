import cv2
import threading
import numpy as np

drawing = False  # 是否开始画图
mode = True  # True：画矩形，False：画圆
mode = 3  # 0:画矩形 1:画圆 2:画点
start = (-1, -1)
img = np.zeros((512, 512, 3), np.uint8)
window_name = 'Image'
regions = []
pts = []


def mouse_event(event, x, y, flags, param):
    global start, drawing, mode, img, regions
    current_img = img.copy()
    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
        if mode == 2:
            cv2.circle(current_img, (x, y), 5, (0, 0, 255), -1)
            pts.append(np.array([x, y]))
            img = current_img.copy()
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode == 0:
                cv2.rectangle(current_img, start, (x, y), (0, 255, 0), 1)
            elif mode == 1:
                cv2.circle(current_img, (x, y), 5, (0, 0, 255), -1)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == 0:
            cv2.rectangle(current_img, start, (x, y), (0, 255, 0), 1)
            width = np.abs(start[0] - x)
            height = np.abs(start[1] - y)
            smaller_ptr = (min(start[0], x), min(start[1], y))
            regions.append((smaller_ptr[0], smaller_ptr[1], width, height))
        elif mode == 1:
            cv2.circle(current_img, (x, y), 5, (0, 0, 255), -1)
            radius = np.sqrt(np.power(start[0] - x, 2) + np.power(start[1] - y, 2))
            regions.append((start[0], start[1], radius))
        img = current_img.copy()
    text = 'x = ' + str(x) + ',' + 'y = ' + str(y)
    cv2.putText(current_img, text, (5, current_img.shape[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow(window_name, current_img)


def select(src, m=2):
    """
    m = 0:选择ROI，返回[(start.x,start.y,end.x,end.y)]形式的数组,表示选定的矩阵区域
    m = 1:选择ROI，返回[centerX,centerY,radius]形式的数组，表示选定的原区域
    m =2 :选定点，返回[(x1,y1),(x2,y2),(x3,y3)....]形式的数组，表示选择的点
    用于简单标定
    :param src:
    :return: regions pointers
    """
    global img, mode
    mode = m
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_event)
    img = src.copy()
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    r = regions.copy()
    pts_cy = pts.copy()
    reset()
    return r, pts_cy


def reset():
    regions.clear()
    pts.clear()
