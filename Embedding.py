import cv2
import numpy as np
import pre
import random
import math
from skimage import metrics

# imread得到的img[x][y][z]，x为行，y为列,cv2.imread读入通道顺序bgr，元素类型为uint8
filename = "test3.tiff"
img_BGR = cv2.imread("./image_data/" + filename)
if img_BGR.ndim == 3:
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)  # BGR转化为YCrCb颜色空间
    img_Y = img_YCrCb[:, :, 0]  # 取Y通道
    # cv2.imshow("img_Y", img_Y)
else:
    img_Y = img_BGR

# pre.write_file(img_Y, "./ori_img_Y.csv")
rho = 1  # 采样率
new_Y = pre.down_sample(img_Y, rho)  # 向下采样，缩小图片
height, width = new_Y.shape

a = 8  # 以关键点为中心的特征区域大小
b = 8  # axb*8x8
# 隐写的水印内容
watermark_string = input("Input a watermark (less than 4 bytes):")
while len(watermark_string) > 6:
    print("Pass the limit!")
    watermark_string = input("Input a watermark (less than 6 bytes):")
watermark_bytes = pre.bch_encode(watermark_string)  # BCH编码

watermark_bin = ""
for ele in watermark_bytes:  # 将字节类型的BCH编码转换为二进制ASCII字符串类型
    ele_s = bin(ele)[2:]
    watermark_bin = watermark_bin + (8 -
                                     len(ele_s)) * '0' + ele_s  # 对单个字母补全8位二进制
watermark_bin = watermark_bin + (a * b -
                                 len(watermark_bin)) * '0'  # 对整个水印序列不全axb的0
watermark_matrix = np.array(list(watermark_bin)).reshape(
    (a, b)).astype("uint8")  # 将水印序列重塑为axb的矩阵
watermark_matrix = watermark_matrix.T  # 水印矩阵按竖列排放
print("---\nwatermark:\n", watermark_bytes, "\n", watermark_bin)

# 计算SIFT关键点，并根据强度排序，即I-SIFT关键点
sigma = 1.6  # 高斯函数sigma参数
gaussian_sigma_k = math.sqrt(2)  # 差分高斯域中的sigma倍数参数k
# kp_list第一个参数为强度，后两个参数为坐标，最后一个为高斯滤波sigma参数
kp_list = pre.get_keypoints(new_Y, sigma, gaussian_sigma_k)
# pre.drawkp(new_Y, kp_list)
# 过滤特征区域超过图像或有重合的关键点
top_n = 10  # top_n强度且不重合符合条件的关键点
side = 8  # side*side的矩阵隐写1位信息
kp_list = pre.kpfilter(kp_list, a, b, height, width, side)
kp_list = kp_list[0:top_n]  # 取符合条件的n个关键点
print("---\nkeypoints: \n", kp_list, "\n---")

# 对n个关键点进行尝试隐写
SSIM_list = []
for kp in kp_list:
    x = kp[1]
    y = kp[2]
    region_x_down, region_x_up = int(x - a * side / 2), int(x + a * side / 2)
    region_y_down, region_y_up = int(y - b * side / 2), int(y + b * side / 2)
    # print(region_x_down, region_x_up)
    # print(region_y_down, region_y_up)
    feature_region = new_Y[region_x_down:region_x_up,
                           region_y_down:region_y_up]
    embed_feature_region = pre.embed_region(feature_region, watermark_matrix,
                                            a, b, side)
    # 最后保存图像的时候需要为整数
    embed_feature_region = embed_feature_region.astype("uint8")
    # 计算n个点关键点的SSIM，只有SSIM值前k个进行隐写
    score = metrics.structural_similarity(feature_region, embed_feature_region)
    # print(score)
    # SSIM_list保存SSIM值、嵌入后区域矩阵、嵌入区域坐标
    SSIM_list.append((score, x, y, embed_feature_region, region_x_down,
                      region_x_up, region_y_down, region_y_up))
    # print("---")
# 选择SSIM值前region_k的关键点进行真正的嵌入
# SSIM_list.sort(reverse=True)
# print("SSIMpoints: \n", SSIM_list, "\n---")
region_k = 5
print("embedSSIMpoints:")
for i in range(min(region_k, len(SSIM_list))):
    new_Y[SSIM_list[i][4]:SSIM_list[i][5],
          SSIM_list[i][6]:SSIM_list[i][7]] = SSIM_list[i][3]
    print(SSIM_list[i][1], SSIM_list[i][2])
img_YCrCb[:, :, 0] = new_Y
new_img = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR)
cv2.imshow("new_img", new_img)
cv2.imwrite("./image_data/" + filename.split('.')[0] + "_embedded" + ".tiff",
            new_img)

pre.drawkp(new_Y, SSIM_list[0:min(region_k, len(SSIM_list))])

# 加框以便reshape
bg_width = 49
frame_width = 1
inside_width = 50 * frame_width
add_shape = bg_width + frame_width + inside_width
frame_img = np.ones((height + 2 * add_shape, width + 2 * add_shape, 3),
                    dtype="uint8")
frame_img *= 255
for x in range(bg_width, height + 2 * add_shape - bg_width):
    frame_img[x, bg_width, :] //= 255
    frame_img[x, width + 2 * add_shape - bg_width - 1, :] //= 255
for y in range(bg_width + 1, width + 2 * add_shape - bg_width - 1):
    frame_img[bg_width, y, :] //= 255
    frame_img[height + 2 * add_shape - bg_width - 1, y, :] //= 255

frame_img[add_shape:height + add_shape,
          add_shape:width + add_shape, :] = new_img
cv2.imshow("frame_img", frame_img)
# pre.write_file(frame_img, "./fff.csv")
cv2.imwrite(
    "./image_data/" + filename.split('.')[0] + "_embedded_frame" + ".tiff",
    frame_img)
cv2.waitKey(0)
