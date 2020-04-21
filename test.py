import cv2
import matplotlib.pyplot as plt
import numpy as np
import bchlib
import os
import random
import math
import pre
import time
'''
# imread得到的img[x][y][z]，x为行，y为列,cv2.imread读入通道顺序bgr
def write_file(img):
    print(img.shape)
    if img.ndim == 2:
        height, width = img.shape
        channel = 1
    if img.ndim == 3:
        height, width, channel = img.shape
    res = ""
    with open("./temp.csv", "w") as f:
        for z in range(channel):
            for x in range(height):
                for y in range(width):
                    if img.ndim == 2:
                        res = res + str(img[x][y]) + ','
                    if img.ndim == 3:
                        res = res + str(img[x][y][z]) + ','
                res = res[:len(res)] + '\n'
                print("\rTotal:" + str(width * height * channel) + "Now:" +
                      str(y + 1 + (x + 1) * width * (z + 1)),
                      end=" ")
                f.write(res)
                res = ""
            f.write('\n\n')
    print("\n")
'''
'''
new_img = np.zeros((height, width, channel), dtype='uint8')
x, y, z = 0, 0, 0
with open("./test.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        if x < height:
            lis = line.split(',')
            for ele in lis:
                new_img[x, y, z] = int(ele)
                y += 1
                if y == width:
                    break
        x += 1
        y = 0
        if x == height + 2:
            x = 0
            z += 1
print(new_img.shape)
cv2.imshow("new", new_img)
'''
'''
ori_img = cv2.imread("./image_data/test.jpeg")
write_file(ori_img)
'''

####################################################################################
'''
BCH_BITS = 4  # BCH能纠正的位数
for BCH_POLYNOMIAL in range(0, 10000):  # 尝试生成多项式的十进制数
    try:
        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
        print(BCH_POLYNOMIAL)
        print(bch.m)  # gf域
        print(bch.n)  # BCH能编码的最大信息长度
        print(bch.ecc_bits)  # ecc所占位数
        print(bch.t)  # BCH能纠正的位数
        print("---")
    except Exception as err:
        continue
'''
'''
BCH_POLYNOMIAL = 285  # 该参数由多项式决定，不能为8的倍数
# 因为当ECC取24bits时，信息最大能取63bits，而该函数只接受bytes类型输入，不能满足63bits要求
# 故使用ECC取32bits，另作规定信息最大取64bits（按隐写区域8x8确定），凑整bytes
# 这个仅为bchlib中该函数实现的问题，matlab中调用bch没有这个问题
BCH_BITS = 4  # BCH能纠正的位数
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
print(bch.m)  # gf域
print(bch.n)  # BCH能编码的最大信息长度bits，超过该长度可能可以编码但不能解码
print(bch.ecc_bits)  # ecc所占位数
print(bch.t)  # BCH能纠正的位数
print("---")
'''
'''
with open('./watermark.txt', 'rb') as f:  # 规定最大4字节
    data = f.read()
'''
'''
string = input("Input a watermark (less than 4 bytes):")
data = bytearray(string.encode())
print(data)
print(type(data))
print(len(data))

ecc = bch.encode(data)
print(ecc)
print(type(ecc))
print(len(ecc))

packet = data + ecc
print(packet)
print(type(packet))
print(len(packet))


def bitflip(packet):
    byte_num = random.randint(0, len(packet) - 1)
    bit_num = random.randint(0, 7)
    packet[byte_num] ^= (1 << bit_num)


# make BCH_BITS errors
for _ in range(BCH_BITS):
    bitflip(packet)

# de-packetize
data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
print("---")
print(packet)
print(type(packet))
print(len(packet))
print("---")
# correct
bitflips = bch.decode_inplace(data, ecc)
print('bitflips: %d' % (bitflips))

# packetize
packet = data + ecc
print(packet)
print(type(packet))
print(len(packet))
'''
'''
def bitflip(packet):
    byte_num = random.randint(0, len(watermark_bytes) - 1)
    bit_num = random.randint(0, 7)
    watermark_bytes[byte_num] ^= (1 << bit_num)


# make BCH_BITS errors
for _ in range(4):
    bitflip(watermark_bytes)
print(watermark_bytes)

watermark_extr = pre.bch_decode(watermark_bytes)
print(watermark_extr)
'''
####################################################################################
'''
# imread得到的img[x][y][z]，x为行，y为列,cv2.imread读入通道顺序bgr，元素类型为uint8
img_BGR = cv2.imread("./image_data/test.jpeg")

img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)  # BGR转化为YCrCb颜色空间
img_Y = img_YCrCb[:, :, 0]  # 取Y通道

width, height = img_Y.shape
rho = 2.5
new_width = int(width // rho)
new_height = int(height // rho)
new_Y = np.zeros((new_width, new_height), dtype="uint8")
for x in range(new_width):
    for y in range(new_height):
        new_Y[x][y] = img_Y[math.ceil(x * rho)][math.ceil(y * rho)]
print(new_Y.shape)
cv2.imshow("new y", new_Y)
cv2.waitKey(0)
'''
####################################################################################
'''
oriimg = cv2.imread("./image_data/test.jpeg")
print("------")
orb = cv2.ORB_create()
print("------")
kp = orb.detect(oriimg)
print("------")
kp, des = orb.compute(oriimg, kp)
print(kp[0])
print(type(kp[0]))
kp.append(cv2.KeyPoint(55, 55, 5))
img = cv2.drawKeypoints(oriimg, kp, None, color=(255, 0, 0))
'''

####################################################################################
'''
def G(x, y, sigma):  # 利用二维高斯函数求权重矩阵
    tmp1 = 1 / (2 * math.pi * math.pow(sigma, 2))
    tmp2 = -(math.pow(x, 2) + math.pow(y, 2)) / (2 * math.pow(sigma, 2))
    tmp3 = tmp1 * math.exp(tmp2)
    return tmp3


def get_L(img_Y, sigma):
    g_shape = 7  # 权重矩阵的维度
    g = np.zeros((g_shape, g_shape))
    cnt = 0
    for x in range(g_shape):
        for y in range(g_shape):
            # 因为高斯模糊是以一个点为中心求周围的权重再反馈给中心点
            # 所以权重矩阵维度只能为奇数，且中心点坐标为(0,0)
            offset = g_shape // 2
            g[x][y] = G(x - offset, y - offset, sigma)
            cnt = cnt + g[x][y]
    
    print(cnt)
    print(g)
    print("-------")
    
    g = g / cnt  # 归一化权重矩阵
    
    cnt = 0
    for x in range(g.shape[0]):
        for y in range(g.shape[1]):
            cnt = cnt + g[x][y]
    print(cnt)
    print(g)
    print("-------")
    
    # 第一个参数为原图像，第二个为深度，-1表示与原图一样，第三个表示卷积核，即归一化后的权重矩阵归一化
    # borderType决定做卷积的时候图像边缘如何处理，默认为None不处理边缘
    # BORDER_CONSTANT填充边缘用指定的像素值：`iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    # BORDER_REPLICATE填充边缘用已知的边缘像素值：`aaaaaa|abcdefgh|hhhhhhh`
    # BORDER_WRAP用另一边的像素值填充：`cdefgh|abcdefgh|abcdefg`
    # BORDER_REFLECT反射复制边界像素：`fedcba|abcdefgh|hgfedcb`
    # BORDER_REFLECT_101以边界为对称轴反射复制像素：`gfedcb|abcdefgh|gfedcba`
    # BORDER_REFLECT101 = BORDER_REFLECT_101
    # BORDER_DEFAULT = BORDER_REFLECT_101
    # BORDER_TRANSPARENT：`uvwxyz|absdefgh|ijklmno`
    # BORDER_ISOLATED： do not look outside of ROI
    res = cv2.filter2D(img_Y.astype("float32"),
                       -1,
                       g,
                       borderType=cv2.BORDER_CONSTANT)
    
    print(res)
    print(res.shape)
    cv2.imshow("222", res)

    res = cv2.GaussianBlur(
        img_Y, (9, 9), sigma,
        borderType=cv2.BORDER_CONSTANT)  # 用论文中的高斯模糊公式+卷积，和该封装函数结果一致
    print(res)
    
    cv2.imshow("111", res)
    cv2.waitKey(0)
    
    return res


img_BGR = cv2.imread("./image_data/test.jpeg")
img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)  # BGR转化为YCrCb颜色空间
img_Y = img_YCrCb[:, :, 0]  # 取Y通道

k = math.sqrt(2)  # 差分高斯域中的倍数参数k
sigma = 1.6 * k  # 高斯函数sigma参数
# L,D每一维，代表k^dim，D要3维3层sigma，k^3*sigma要用到k^4*sigma来求，故L要4层即k^4*sigma
# D特征点是以中心点的3x3x3来求的，只需要3维3层sigma，I为D的绝对值
L = np.zeros((img_Y.shape[0], img_Y.shape[1], 4))
D = np.zeros((img_Y.shape[0], img_Y.shape[1], 3))
L_x, L_y, L_dim = L.shape
D_x, D_y, D_dim = D.shape

start = time.time()
# 计算高斯模糊
for dim in range(L_dim):
    res_L = get_L(img_Y, math.pow(k, dim) * sigma)
    L[:, :, dim] = res_L
    
    #for x in range(L_x):
    #    for y in range(L_y):
    #        L[x][y][dim] = res_L[x][y]
    
# print(L)
# print("----")
# 计算差分高斯域的绝对值即强度II_dim
for dim in range(D_dim):
    D[:, :, dim] = L[:, :, dim + 1] - L[:, :, dim]
    
    #for x in range(D_x):
    #    for y in range(D_y):
    #        D[x][y][dim] = float(L[x][y][dim + 1]) - float(L[x][y][dim])
    
end = time.time()
print(round(end - start, 2))

start = time.time()
# 找到所有极值点并排序
extreme_points_list = []  # 点列表，一个元组代表一个点，第一位为该点强度，后三位为该点坐标
for dim in range(1, D_dim - 1):
    print("-----")
    print(dim)
    start1 = time.time()
    for x in range(1, D_x - 1):
        for y in range(1, D_y - 1):
            matrix_max = D[(x - 1):(x + 2), (y - 1):(y + 2),
                           (dim - 1):(dim + 2)].max()
            matrix_min = D[(x - 1):(x + 2), (y - 1):(y + 2),
                           (dim - 1):(dim + 2)].min()
            value = D[x][y][dim]
            if (value == matrix_max) or (value == matrix_min):
                extreme_points_list.append((abs(value), x, y, dim))
    end1 = time.time()
    print(round(end1 - start1, 2))
end = time.time()
print(round(end - start, 2))

points_list = []
for item in extreme_points_list:
    # 过滤低对比度，阈值取经验值0.03/0.04
    # https://blog.csdn.net/lingyunxianhe/article/details/79063547
    if item[0] > 0.03:
        # 去除不稳定噪声点，对D求导和Hessian矩阵计算结果对于阈值公式过滤
        # 导数可由采样点相邻差估计得到？
        # https://blog.csdn.net/wd1603926823/article/details/46453629
        # https://blog.csdn.net/abcjennifer/article/details/7639681
        dxx = D[item[1]][item[2] + 1][item[3]] + \
                D[item[1]][item[2] - 1][item[3]] - \
                2 * D[item[1]][item[2]][item[3]]
        dyy = D[item[1] + 1][item[2]][item[3]] + \
                D[item[1] - 1][item[2]][item[3]] - \
                2 * D[item[1]][item[2]][item[3]]
        dxy = D[item[1] + 1][item[2] + 1][item[3]] - \
                D[item[1] + 1][item[2] - 1][item[3]] - \
                D[item[1] - 1][item[2] + 1][item[3]] + \
                D[item[1] - 1][item[2] - 1][item[3]]
        tr = dxx + dyy
        det = dxx * dyy - math.pow(dxy, 2)
        r = 10  # Lowe论文中取10
        if math.pow(tr, 2) / det < math.pow(r + 1, 2) / 2:
            points_list.append(item)

points_list.sort(reverse=True)
print(len(points_list))
with open("./ttt.csv", "w") as f:
    for ele in points_list:
        f.write(str(ele))
        f.write("\n")
'''
####################################################################################
'''
img = cv2.imread("./image_data/test_shooting.jpg")

rho = 5
new_img = pre.down_sample(img, rho)
img_Y = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
print(img_Y.shape)

# 高斯模糊降噪模糊
img_Y = cv2.GaussianBlur(img_Y, (9, 9), 1.6, borderType=cv2.BORDER_DEFAULT)
img_Y = pre.get_L(img_Y, 1.6, cv2.BORDER_DEFAULT).astype("uint8")
cv2.imshow("img_Y", img_Y)
cv2.waitKey()

img_Y = cv2.Canny(img_Y, 50, 150)
cv2.imshow("lane", img_Y)
# print(img_Y.shape)
#pre.write_file(img_Y)

cv2.waitKey()
'''
####################################################################################

img = cv2.imread("./image_data/test3_shoot3.jpg")
'''
height, width, channel = img.shape
new = np.ones((height + 100, width + 100, channel), dtype="uint8")
new = new * 255

print(new[50:height + 50, 50:width + 50, :].shape)
new[50:height + 50, 50:width + 50, :] = img
print(new.shape)
cv2.imshow("img", new)
cv2.imwrite("./image_data/test3_simulate_screen.bmp", new)
cv2.waitKey(0)
'''
pre.write_file(img, "./test.csv")
####################################################################################
