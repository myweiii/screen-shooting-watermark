import pre
import cv2
import numpy as np
import math

filename = "test3_shoot5.jpg"
img = cv2.imread("./image_data/" + filename)

rho = 1
new_img = pre.down_sample(img, rho)

frame_width = 1
inside_width = 50 * frame_width
add_shape = frame_width + inside_width
ori_height = 512 + 2 * add_shape
ori_width = 512 + 2 * add_shape

if filename.find("screen") > 0 or filename.find("shoot") > 0 or filename.find(
        "frame") > 0:
    sigma = 1.6
    img_reshape = pre.get_reshape(
        new_img, ori_height, ori_width,
        sigma)[(frame_width + inside_width):(ori_height - add_shape),
               (frame_width + inside_width):(ori_width - add_shape), :]
    cv2.imshow("img_reshape", img_reshape)
    img_Y = cv2.cvtColor(img_reshape, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    print("---\nreshape: \n", img_Y.shape)
else:
    img_Y = cv2.cvtColor(new_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    # pre.write_file(img_Y, "./screen_Y.csv")
    # cv2.imshow("img_Y", img_Y)
'''
img_Y = cv2.cvtColor(new_img, cv2.COLOR_BGR2YCrCb)[:, :, 0][25:537, 33:545]
cv2.imshow("img_Y", img_Y)
'''
a = 8  # 以关键点为中心的特征区域大小
b = 8  # axb*8x8
side = 8
# 计算SIFT关键点，并根据强度排序，即I-SIFT关键点
sigma = 1.6  # 高斯函数sigma参数
gaussian_sigma_k = math.sqrt(2)  # 差分高斯域中的sigma倍数参数k
# kp_list第一个参数为强度，后两个参数为坐标，最后一个为高斯滤波sigma参数
kp_list = pre.get_keypoints(img_Y, sigma, gaussian_sigma_k)

# 过滤特征区域超过图像或有重合的关键点
region_k = 5
top_n = 2 * region_k  # top_n强度且不重合符合条件的关键点
side = 8  # side*side的矩阵隐写1位信息
kp_list = pre.kpfilter_extract(kp_list)
kp_list = kp_list[0:top_n]

kp_list.append((0, 426, 163, 0))
kp_list.append((0, 373, 385, 0))
kp_list.append((0, 98, 83, 0))
kp_list.append((0, 178, 34, 0))
kp_list.append((0, 243, 91, 0))

print("---\nkeypoints: \n", kp_list, "\n---")
pre.drawkp(img_Y, kp_list)

res_list = []
for kp in kp_list:
    x = kp[1]
    y = kp[2]
    res = pre.get_watermark_group(img_Y, x, y, a, b, side)
    res_list.append(res)
# print(res_list)
th = 6
watermark_bin, watermark_bin_2 = pre.cross_validation(res_list, a, b, th,
                                                      kp_list)
print("watermark_bin:\nresult1: ", watermark_bin)
print("result2: ", watermark_bin_2, "\n---")

try:
    print("result1: ", pre.watermark_bin_decode_reshape(watermark_bin))
    print("result2: ", pre.watermark_bin_decode_reshape(watermark_bin_2))
except Exception as err:
    print("Can't decode! Err: ", err)
