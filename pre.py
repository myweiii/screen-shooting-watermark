import cv2
import numpy as np
import bchlib
import os
import math
import struct
from numpy.linalg import solve


def write_file(img, filename):  # 写进csv方便查看变量
    print(img.shape)
    sh = img.shape
    res = ""
    with open(filename, "w") as f:
        for x in range(sh[0]):
            for y in range(sh[1]):
                if img.ndim == 2:
                    res = res + str(img[x][y]) + ','
                if img.ndim == 3:
                    res = res + "( " + str(img[x][y][0]) + " " + str(
                        img[x][y][1]) + " " + str(img[x][y][2]) + " )" + ','
            res = res[:len(res)] + '\n'
            print("\rTotal:" + str(sh[0] * sh[1]) + "Now:" +
                  str(y + 1 + (x + 1) * sh[1]),
                  end=" ")
            f.write(res)
            res = ""
        f.write('\n\n')
    print("\n")


def bch_encode(string):  # bch编码，接受string类型输入
    BCH_POLYNOMIAL = 285  # 该参数由多项式决定，不能为8的倍数
    # 因为当ECC取24bits时，信息最大能取63bits，而该函数只接受bytes类型输入，不能满足63bits要求
    # 故使用ECC取32bits，另作规定信息最大取80bits（按隐写区域9x9确定，1位冗余），凑整bytes
    # 即水印大小最大可为48bits，即6bytes
    # 这个仅为bchlib中该函数实现的问题，matlab中调用bch没有这个问题
    BCH_BITS = 4  # BCH能纠正的位数
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(string.encode())
    ecc = bch.encode(data)
    packet = data + ecc

    return packet


def bch_decode(packet):  # bch解码纠错，接受bytes类型输入
    BCH_POLYNOMIAL = 285  # 该参数由多项式决定，不能为8的倍数
    # 因为当ECC取24bits时，信息最大能取63bits，而该函数只接受bytes类型输入，不能满足63bits要求
    # 故使用ECC取32bits，另作规定信息最大取64bits（按隐写区域8x8确定），凑整bytes
    # 这个仅为bchlib中该函数实现的问题，matlab中调用bch没有这个问题
    BCH_BITS = 4  # BCH能纠正的位数
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
    bch.decode_inplace(data, ecc)
    packet = data + ecc

    return packet


def down_sample(img_Y, rho):  # 向下采样，缩小图片大小，rho为缩小倍数
    if img_Y.ndim == 2:
        width, height = img_Y.shape
        new_width = int(width // rho)
        new_height = int(height // rho)
        new_Y = np.zeros((new_width, new_height), dtype="uint8")
        for x in range(new_width):
            for y in range(new_height):
                new_Y[x][y] = img_Y[math.ceil(x * rho)][math.ceil(y * rho)]
    else:
        new_Y0 = down_sample(img_Y[:, :, 0], rho)
        new_Y1 = down_sample(img_Y[:, :, 1], rho)
        new_Y2 = down_sample(img_Y[:, :, 2], rho)
        new_Y = np.zeros((new_Y0.shape[0], new_Y0.shape[1], 3), dtype="uint8")
        new_Y[:, :, 0] = new_Y0
        new_Y[:, :, 1] = new_Y1
        new_Y[:, :, 2] = new_Y2
    return new_Y


def drawkp(img_Y, kp_list):  # 在图上圈出关键点位置
    kp = []
    for k in kp_list:
        kp.append(cv2.KeyPoint(k[2], k[1], 1))
    img = cv2.drawKeypoints(img_Y, kp, None, color=(255, 0, 0))
    cv2.imshow("draw_keypoints", img)
    cv2.waitKey(0)


def G(x, y, sigma):  # 利用二维高斯函数求权重矩阵
    tmp1 = 1 / (2 * math.pi * math.pow(sigma, 2))
    tmp2 = -(math.pow(x, 2) + math.pow(y, 2)) / (2 * math.pow(sigma, 2))
    tmp3 = tmp1 * math.exp(tmp2)
    return tmp3


def get_L(img_Y, sigma, type):  # 计算高斯模糊，type决定卷积边缘处理模式
    g_shape = 9  # 权重矩阵的二维度，权重矩阵以点为中心的矩阵
    g = np.zeros((g_shape, g_shape), dtype="float32")
    cnt = 0
    for x in range(g_shape):
        for y in range(g_shape):
            # 因为高斯模糊是以一个点为中心求周围的权重再反馈给中心点
            # 所以权重矩阵维度只能为奇数，且中心点坐标为(0,0)
            offset = g_shape // 2
            g[x][y] = G(x - offset, y - offset, sigma)
            cnt = cnt + g[x][y]
    g = g / cnt  # 归一化权重矩阵
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
    res = cv2.filter2D(img_Y.astype("float32"), -1, g, borderType=type)
    '''
    res = cv2.GaussianBlur(img_Y.astype("float32"), (9, 9), sigma,borderType=type)  # 用论文中的高斯模糊公式+卷积，和该封装函数结果大致一致
    '''
    return res


def get_keypoints(img_Y, ini_sigma, k):  # 提取关键点
    sigma = ini_sigma * k  # 高斯函数sigma参数
    # L,D每一维，代表k^dim，D要3维3层sigma，k^3*sigma要用到k^4*sigma来求，故L要4层即k^4*sigma
    # D特征点是以中心点的3x3x3来求的，只需要3维3层sigma，I为D的绝对值
    L = np.zeros((img_Y.shape[0], img_Y.shape[1], 4), dtype="float32")
    D = np.zeros((img_Y.shape[0], img_Y.shape[1], 3), dtype="float32")
    L_x, L_y, L_dim = L.shape
    D_x, D_y, D_dim = D.shape

    # 计算高斯模糊
    for dim in range(L_dim):
        res_L = get_L(img_Y, math.pow(k, dim) * sigma, cv2.BORDER_CONSTANT)
        L[:, :, dim] = res_L
    # 计算差分高斯域的绝对值即强度II_dim
    for dim in range(D_dim):
        D[:, :, dim] = L[:, :, dim + 1] - L[:, :, dim]

    # 找到所有极值点并排序
    extreme_points_list = []  # 点列表，一个元组代表一个点，第一位为该点强度，后三位为该点坐标
    for dim in range(1, D_dim - 1):
        for x in range(1, D_x - 1):
            for y in range(1, D_y - 1):
                matrix_max = D[(x - 1):(x + 2), (y - 1):(y + 2),
                               (dim - 1):(dim + 2)].max()
                matrix_min = D[(x - 1):(x + 2), (y - 1):(y + 2),
                               (dim - 1):(dim + 2)].min()
                value = D[x][y][dim]
                if (value == matrix_max) or (value == matrix_min):
                    extreme_points_list.append(
                        (abs(value), x, y, dim))  # 取D绝对值作为强度
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
            if det != 0 and math.pow(tr, 2) / det < math.pow(r + 1, 2) / 2:
                points_list.append(item)

    points_list.sort(reverse=True)
    # print(len(points_list))
    return points_list


def kpfilter(kps, a, b, height, width, side):  # 过滤有重复区域的关键点
    # 一个关键点占区域axb*side*side
    # 这里ab取9，一个关键点占区域72*72
    rm_list = []
    for idx in range(0, len(kps)):
        x = kps[idx][1]  # x-a*side/2:x+a*side/2
        y = kps[idx][2]  # y-b*side/2:y+b*side/2
        if x - a * side / 2 < 0 or x + a * side / 2 >= height or y - b * side / 2 < 0 or y + b * side / 2 >= width:  # 特征区域超出图像范围
            rm_list.append(idx)
            continue

        for other_idx in range(idx + 1, len(kps)):
            if other_idx in rm_list:
                continue
            other_x = kps[other_idx][1]
            other_y = kps[other_idx][2]
            if (((x - a * side / 2) <= (other_x - a * side / 2) <=
                 (x + a * side / 2)) or
                ((x - a * side / 2) <= (other_x + a * side / 2) <=
                 (x + a * side / 2))) and (((y - b * side / 2) <=
                                            (other_y - a * side / 2) <=
                                            (y + b * side / 2)) or
                                           ((y - b * side / 2) <=
                                            (other_y + a * side / 2) <=
                                            (y + b * side / 2))):  # 特征区域有重叠
                rm_list.append(other_idx)

    rm_list = list(set(rm_list))
    rm_list.sort()

    for i in range(0, len(rm_list)):  # 去掉特征区域不符合的特征点
        kps.pop(rm_list[i] - i)
    return kps


def kpfilter_extract(kps):  # 过滤有重复区域的关键点
    # 一个关键点占区域axb*side*side
    # 这里ab取9，一个关键点占区域72*72
    rm_list = []
    for idx in range(0, len(kps)):
        x = kps[idx][1]  # x-a*side/2:x+a*side/2
        y = kps[idx][2]  # y-b*side/2:y+b*side/2

        for other_idx in range(idx + 1, len(kps)):
            if other_idx in rm_list:
                continue
            other_x = kps[other_idx][1]
            other_y = kps[other_idx][2]
            if (((x - 1) <= (other_x - 1) <= (x + 1)) or
                ((x - 1) <= (other_x + 1) <=
                 (x + 1))) and (((y - 1) <= (other_y - 1) <=
                                 (y + 1)) or ((y - 1) <= (other_y + 1) <=
                                              (y + 1))):  # 特征区域有重叠
                rm_list.append(other_idx)

    rm_list = list(set(rm_list))
    rm_list.sort()

    for i in range(0, len(rm_list)):  # 去掉特征区域不符合的特征点
        kps.pop(rm_list[i] - i)
    return kps


def embed_region(feature_region, watermark_matrix, a, b, side):  # 对于特征区域进行嵌入
    # 调用DCT变换方法需要接受浮点数类型
    feature_region = feature_region.astype("float32")
    d = 512  # C1，C2之差的阈值，防止jpg压缩产生变化，可由公式得出
    for block_y in range(0, b):  # 对于每一个嵌入块，以(4,5)和(5,4)值的相对大小隐写
        for block_x in range(0, a):  # 竖排方向嵌入
            block = feature_region[block_x * side:(block_x * side + side),
                                   block_y * side:(block_y * side + side)]
            block_DCT = cv2.dct(block)
            C1_x, C1_y = int(side / 2 + 1), int(side / 2)  # 这里的xy和论文中的xy是反的
            C2_x, C2_y = int(side / 2), int(side / 2 + 1)
            C1 = block_DCT[C1_x][C1_y]
            C2 = block_DCT[C2_x][C2_y]
            max_C = max(C1, C2)
            min_C = min(C1, C2)
            if watermark_matrix[block_x][block_y] == 0:  # C1>C2嵌入0，C1<C2嵌入1
                block_DCT[C1_x][C1_y] = max_C + d / 2
                block_DCT[C2_x][C2_y] = min_C - d / 2
            else:
                block_DCT[C1_x][C1_y] = min_C - d / 2
                block_DCT[C2_x][C2_y] = max_C + d / 2
            block = cv2.idct(block_DCT)
            feature_region[block_x * side:(block_x * side + side), block_y *
                           side:(block_y * side + side)] = block[:, :]
    return feature_region


def side_trace(img, ini_x, y, height, angle1, angle2):  # 跟踪边找顶点
    # angle1-1向上，1向下，angle2-1向左，1向右
    final_x, final_y = ini_x, y
    for i in range(1, (height - ini_x)):
        x = ini_x + i * angle1
        cnt = 0
        new_y = y
        if img[x][y + 1 * angle2] == 255:
            new_y = y + 1 * angle2
            cnt += 1
        if img[x][y] == 255:
            new_y = y
            cnt += 1
        if img[x][y - 1 * angle2] == 255:
            new_y = y - 1 * angle2
            cnt += 1
        if cnt == 0 and img[x + 1 * angle1][y + 1 * angle2] == img[
                x + 1 * angle1][y] == img[x + 1 * angle1][y - 1 * angle2] == 0:
            final_x, final_y = x - 1 * angle1, y
            break
        y = new_y
        if cnt >= 2 and img[x + 1 * angle1][y + 1 * angle2] == img[
                x + 1 * angle1][y] == img[x + 1 * angle1][
                    y - 1 * angle2] == 0 and img[x + 2 * angle1][
                        y + 1 * angle2] == img[x + 2 * angle1][y] == img[
                            x + 2 * angle1][y - 1 * angle2] == 0:
            final_x, final_y = x, y
            break
    return (final_x, final_y)


def get_4_corners(img, f):  # 找到4个顶点
    height, width = img.shape
    x = height // 2  # 从中线开始找

    for y in range(width):
        if img[x][y] == 255:
            (cor_x1, cor_y1) = side_trace(img, x, y, height, -1, 1)  # 从左到右向上找
            (cor_x2, cor_y2) = side_trace(img, x, y, height, 1, 1)  # 从左到右向下找
            if abs(cor_x1 - x) + abs(cor_x2 - x) > f:  # 超过该阈值认为是图片边框
                break
        # img[x][y] = 255

    for i in range(1, width):
        y = width - i
        if img[x][y] == 255:
            (cor_x3, cor_y3) = side_trace(img, x, y, height, -1, -1)  # 从右到左向上找
            (cor_x4, cor_y4) = side_trace(img, x, y, height, 1, -1)  # 从右到左向下找
            if abs(cor_x3 - x) + abs(cor_x4 - x) > f:
                break
        # img[x][y] = 255

    return (cor_x1, cor_y1, cor_x2, cor_y2, cor_x3, cor_y3, cor_x4, cor_y4)


def get_ori_pos(ori_x, ori_y, a0, a1, a2, b0, b1, b2, c1, c2):
    # 根据映射公式得到坐标
    camera_x = (a1 * ori_x + b1 * ori_y + c1) / (a0 * ori_x + b0 * ori_y + 1)
    camera_y = (a2 * ori_x + b2 * ori_y + c2) / (a0 * ori_x + b0 * ori_y + 1)
    return (int(camera_x + 0.5), int(camera_y + 0.5))  # 返回四舍五入值


def get_reshape(camera_img, ori_height, ori_width, sigma):  # 将图像重整为原大小
    f = ori_height // 2  # 判断是否为图像边缘阈值
    # 高斯模糊降噪模糊
    # img = cv2.GaussianBlur(img, (9, 9), sigma, borderType=cv2.BORDER_DEFAULT)
    img = (get_L(camera_img, sigma, cv2.BORDER_DEFAULT) + 0.5).astype("uint8")
    # cv2.imshow("img", img)

    # canny边缘检测，第二个参数最小阈值，第三个最大阈值
    # 进行高斯滤波平滑图像
    # 图像中边缘可以指向任何方向，计算图像的梯度和方向，并将梯度分类为垂直水平和斜对角
    # 检测某一像素在梯度的正方向和反方向上受否为局部最大值，如是则保留为边缘点
    # 仍然存在由于噪声和颜色变化引起的一些边缘像素、
    # 为解决杂散效应，需要用地梯度过滤边缘像素，保留高梯度边缘像素，通过阈值设定

    img = cv2.Canny(img, 50, 150)
    # cv2.imshow("canny_img", img)
    # write_file(img, "./canny.csv")
    cor_x1, cor_y1, cor_x2, cor_y2, cor_x3, cor_y3, cor_x4, cor_y4 = get_4_corners(
        img, f)
    # cor_x1, cor_y1, cor_x2, cor_y2, cor_x3, cor_y3, cor_x4, cor_y4 = 442, 182, 1075, 183, 442, 821, 1081, 817
    # cor_x1, cor_y1, cor_x2, cor_y2, cor_x3, cor_y3, cor_x4, cor_y4 = 1324, 544, 3225, 549, 1324, 2468, 3242, 2451
    print("---\ncorners:\nx1: ", cor_x1, cor_y1, "\nx2: ", cor_x2, cor_y2,
          "\nx3: ", cor_x3, cor_y3, "\nx4: ", cor_x4, cor_y4)

    kp = []
    kp.append(cv2.KeyPoint(cor_y1, cor_x1, 1))
    kp.append(cv2.KeyPoint(cor_y2, cor_x2, 1))
    kp.append(cv2.KeyPoint(cor_y3, cor_x3, 1))
    kp.append(cv2.KeyPoint(cor_y4, cor_x4, 1))
    img = cv2.drawKeypoints(img, kp, None, color=(255, 255, 255))
    cv2.namedWindow("draw_corners", 0)
    cv2.imshow("draw_corners", img)

    ori_x1, ori_y1, ori_x2, ori_y2 = 0, 0, ori_height - 1, 0
    ori_x3, ori_y3, ori_x4, ori_y4 = 0, ori_width - 1, ori_height - 1, ori_width - 1
    # 用矩阵计算方程组
    eq1_1 = [ori_x1 * cor_x1, -ori_x1, 0, ori_y1 * cor_x1, -ori_y1, 0, -1, 0]
    eq2_1 = [ori_x1 * cor_y1, 0, -ori_x1, ori_y1 * cor_y1, 0, -ori_y1, 0, -1]
    eq1_2 = [ori_x2 * cor_x2, -ori_x2, 0, ori_y2 * cor_x2, -ori_y2, 0, -1, 0]
    eq2_2 = [ori_x2 * cor_y2, 0, -ori_x2, ori_y2 * cor_y2, 0, -ori_y2, 0, -1]
    eq1_3 = [ori_x3 * cor_x3, -ori_x3, 0, ori_y3 * cor_x3, -ori_y3, 0, -1, 0]
    eq2_3 = [ori_x3 * cor_y3, 0, -ori_x3, ori_y3 * cor_y3, 0, -ori_y3, 0, -1]
    eq1_4 = [ori_x4 * cor_x4, -ori_x4, 0, ori_y4 * cor_x4, -ori_y4, 0, -1, 0]
    eq2_4 = [ori_x4 * cor_y4, 0, -ori_x4, ori_y4 * cor_y4, 0, -ori_y4, 0, -1]
    eq = np.mat([eq1_1, eq2_1, eq1_2, eq2_2, eq1_3, eq2_3, eq1_4, eq2_4])

    r1_1, r2_1 = -cor_x1, -cor_y1
    r1_2, r2_2 = -cor_x2, -cor_y2
    r1_3, r2_3 = -cor_x3, -cor_y3
    r1_4, r2_4 = -cor_x4, -cor_y4
    r = np.mat([r1_1, r2_1, r1_2, r2_2, r1_3, r2_3, r1_4, r2_4]).T
    # print(r)
    # 解方程组的到8个参数
    res = solve(eq, r)

    a0, a1, a2, b0, b1, b2, c1, c2 = res
    a0, a1, a2, b0, b1, b2, c1, c2 = float(a0), float(a1), float(a2), float(
        b0), float(b1), float(b2), float(c1), float(c2)
    # print(a0, a1, a2, b0, b1, b2, c1, c2)

    img_reshape = np.zeros((ori_height, ori_width, 3), dtype="uint8")
    for x in range(ori_height):
        for y in range(ori_width):
            camera_x, camera_y = get_ori_pos(x, y, a0, a1, a2, b0, b1, b2, c1,
                                             c2)
            img_reshape[x, y, :] = camera_img[camera_x, camera_y, :]
    # cv2.imshow("reshape", img_reshape)

    # cv2.waitKey()
    return img_reshape


def get_watermark_group(img_Y, ini_x, ini_y, a, b, side):  # 获取一个关键点的水印组
    cnt = 0
    res_group = []
    for x in range(ini_x - 1, ini_x + 2):  # 针对关键点周围的9个点，为一水印组
        for y in range(ini_y - 1, ini_y + 2):
            cnt += 1
            region_x_down, region_x_up = int(x -
                                             a * side / 2), int(x +
                                                                a * side / 2)
            region_y_down, region_y_up = int(y -
                                             b * side / 2), int(y +
                                                                b * side / 2)
            if region_x_down < 0 or region_x_up > img_Y.shape[
                    0] or region_y_down < 0 or region_y_up > img_Y.shape[1]:
                continue

            region = img_Y[region_x_down:region_x_up,
                           region_y_down:region_y_up]
            res = []
            for block_y in range(0, b):  # 对于每一个嵌入块，以(4,5)和(5,4)值的相对大小隐写
                for block_x in range(0, a):
                    block = region[block_x * side:(block_x * side + side),
                                   block_y * side:(block_y * side + side)]
                    block_DCT = cv2.dct(block.astype("float32"))
                    C1_x, C1_y = int(side / 2 + 1), int(side / 2)  # (4,5)
                    C2_x, C2_y = int(side / 2), int(side / 2 + 1)  # (5,4)
                    C1 = block_DCT[C1_x][C1_y]
                    C2 = block_DCT[C2_x][C2_y]
                    if C1 >= C2:
                        bit = 0
                    else:
                        bit = 1
                    res.append(bit)
            res_group.append(res)
            # print(cnt, block.shape)
    return res_group


def cross_validation(watermark_group_list, a, b, th, kp_list):
    watermark_pair_list = []
    # 一个水印组内的每一条水印都与其他组中的每一条水印进行配对，组成水印对
    print("right points:")
    rpi = []
    for now_group in range(len(watermark_group_list)):
        now_watermark_group = watermark_group_list[now_group]
        for index_now_wm in range(len(now_watermark_group)):
            now_watermark = now_watermark_group[index_now_wm]
            for other_group in range(now_group, len(watermark_group_list)):
                if other_group == now_group:
                    continue
                other_watermark_group = watermark_group_list[other_group]
                for index_other_wm in range(len(other_watermark_group)):
                    other_watermark = other_watermark_group[index_other_wm]
                    diff = 0
                    for idx in range(a * b):
                        diff = diff + (now_watermark[idx]
                                       ^ other_watermark[idx])
                    if diff <= th:  # 如果水印对中一对水印的位数差距小于th，则认为相似，加入水印对序列
                        watermark_pair_list.append(now_watermark)
                        watermark_pair_list.append(other_watermark)
                        rpi.append((now_group, other_group))
    for item in set(rpi):
        print("index: ", item, "\npoints: ", kp_list[item[0]],
              kp_list[item[1]])
    print("---")
    # print(len(watermark_pair_list))

    watermark_pair_list = np.array(watermark_pair_list)
    # print(watermark_pair_list)
    res = np.zeros(a * b, dtype="uint8")
    l = len(watermark_pair_list)  # 水印对序列的长度
    # 水印对序列中每一条水印的对应位相加，如果该位和超过总长一半则认为该位为1，否则为0
    # 类似投票制
    for watermark in watermark_pair_list:
        res = res + watermark
    print("vote result: ( l = ", l, ")\n", res, "\n---")
    watermark_bin = ""
    watermark_bin_2 = ""
    for bit in res:
        if bit >= l / 2:
            watermark_bin = watermark_bin + "1"
            watermark_bin_2 = watermark_bin_2 + "0"
        else:
            watermark_bin = watermark_bin + "0"
            watermark_bin_2 = watermark_bin_2 + "1"
    return watermark_bin, watermark_bin_2


def watermark_bin_decode_reshape(watermark_bin):
    watermark_bytes = b''
    for byte_idx in range(len(watermark_bin) // 8):
        bin_byte = watermark_bin[byte_idx * 8:byte_idx * 8 + 8]
        int_byte = int(bin_byte, 2)
        #if byte_idx == 0:
        #    int_byte += 1
        watermark_bytes = watermark_bytes + struct.pack('B', int_byte)

    up = len(watermark_bytes)
    for idx in range(1, len(watermark_bytes)):
        if watermark_bytes[len(watermark_bytes) - idx] == 0:
            up = len(watermark_bytes) - idx
        else:
            break
    watermark_bytes = watermark_bytes[0:up]
    watermark = bch_decode(bytearray(watermark_bytes))
    return watermark
