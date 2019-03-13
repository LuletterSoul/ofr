import cv2
import numpy as np
import util.PlotUtil as plot
import os

plot_index = 0;


def inc():
    global plot_index
    plot_index += 1
    return plot_index


def reset():
    global plot_index
    plot_index = 0


img = cv2.imread('image/IMG_9064.JPG')
img = cv2.GaussianBlur(img, (3, 3))
plot.subImage(src=cv2.cvtColor(img, cv2.COLOR_RGB2BGR), index=inc(), title='Source', cmap='gray')

kernel_2 = np.ones((2, 2), np.uint8)  # 2x2的卷积核

kernel_3 = np.ones((3, 3), np.uint8)  # 3x3的卷积核

kernel_4 = np.ones((4, 4), np.uint8)  # 4x4的卷积核

minValues = np.array([26, 43, 46])
maxValues = np.array([34, 255, 255])
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_hsv, minValues, maxValues)

img_yellow_org = cv2.bitwise_and(img, img, mask=mask)
plot.subImage(src=cv2.cvtColor(img_yellow_org, cv2.COLOR_RGB2BGR), index=inc(), title='Org Yellow')
erosion = cv2.erode(mask, kernel_2, iterations=1)
erosion = cv2.erode(erosion, kernel_2, iterations=1)
dilation = cv2.dilate(erosion, kernel_2, iterations=1)
dilation = cv2.dilate(dilation, kernel_2, iterations=1)
img_yellow = cv2.bitwise_and(img, img, mask=dilation)

plot.subImage(src=cv2.cvtColor(img_yellow_org, cv2.COLOR_RGB2BGR), index=inc(), title='Extract Yellow Color')
gray = cv2.cvtColor(img_yellow, cv2.COLOR_RGB2GRAY)
retval, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plot.subImage(src=binary, index=inc(), title='Binary', cmap='gray')

binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if len(c) > 150]
rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
cv2.drawContours(rgb, contours, -1, (0, 0, 255), 2)
angles = [cv2.minAreaRect(c)[2] for c in contours]
his = np.histogram(angles, 9, [-90, 0])
low45 = np.sum(his[0][0:4])
high45 = np.sum(his[0][5:])
print("Low - High his:", [low45, high45])
print(his)
angles = 0
for contour in contours:
    rect = cv2.minAreaRect(contour)
    angles += rect[2]
    pts = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(rgb, [pts], -1, (0, 0, 255), 3)
if len(contours) > 0:
    print(len(contours))
    print("Average angle sum :", angles / len(contours))
plot.subImage(src=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), index=inc(), title="Contours")
plot.save()
