import cv2
import numpy as np

im = cv2.imread("bear.jpg", 1)

# 将图片转成YCbCr型
imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# imYCBCR = cv2.cvtColor(imRGB, cv2.COLOR_RGB2YCR_CB)
# YY = imYCBCR[:, :, 0]
# Cb = imYCBCR[:, :, 2]
# Cr = imYCBCR[:, :, 1]
# m, n, c = imYCBCR.shape
# tst = np.zeros([m, n])
# Mb = np.mean(np.mean(Cb))
# Mr = np.mean(np.mean(Cr))
#
# # 计算Cb, Cr的均方差
# Tb = Cb - Mb
# Tr = Cr - Mr
#
# Db = np.sum(np.sum(Tb * Tb)) / (m * n)
# Dr = np.sum(np.sum(Tr * Tr)) / (m * n)
#
# # 根据阈值的要求提取出near - white区域的像素点
# cnt = 0
# Ciny = np.zeros([m * n])
# for i in range(m):
#     for j in range(n):
#         b1 = Cb[i, j] - (Mb + np.dot(Db, np.sign(Mb)))
#         b2 = Cr[i, j] - (1.5 * Mr + np.dot(Dr, np.sign(Mr)))
#         if b1 < np.abs(1.5 * Db) and b2 < np.abs(1.5 * Dr):
#             Ciny[cnt] = YY[i, j]
#             tst[i, j] = YY[i, j]
#             cnt += 1
#
# cnt -= 1
# iy = sorted(Ciny,reverse=True)
# nn = round(cnt / 10)
# Ciny2 = []
# Ciny2[0:nn] = iy[0:nn]
# mn = min(Ciny2)
# for i in range(m):
#     for j in range(n):
#         if tst[i, j] < mn:
#             tst[i, j] = 0
#         else:
#             tst[i, j] = 1
#
# R = imRGB[:, :, 0]
# G = imRGB[:, :, 1]
# B = imRGB[:, :, 2]
#
# R = np.double(R) * tst
# G = np.double(G) * tst
# B = np.double(B) * tst
#
# Rav = np.mean(np.mean(R))
# Gav = np.mean(np.mean(G))
# Bav = np.mean(np.mean(B))
#
# Ymax = np.double(np.max(np.max(YY))) * 0.15
#
# Rgain = Ymax / Rav
# Ggain = Ymax / Gav
# Bgain = Ymax / Bav
#
# new_im = np.zeros([m, n, 3])
# new_im[:, :, 0] = imRGB[:, :, 0] * Rgain
# new_im[:, :, 1] = imRGB[:, :, 1] * Ggain
# new_im[:, :, 2] = imRGB[:, :, 2] * Bgain
#
# cv2.imshow('img', new_im)
# cv2.waitKey(0)
choice = 3
if choice == 0:
    dst = cv2.bilateralFilter(im, 15, 35, 35)
    m, n, c = im.shape[0:3]
if choice == 1:
    (b, g, r) = cv2.split(im)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    dst = cv2.merge((bH, gH, rH), )

if choice == 2:
    imgYUV = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    channelsYUV = cv2.split(imgYUV)
    channelsYUV0 = cv2.equalizeHist(channelsYUV[0])
    channels = cv2.merge((channelsYUV0, channelsYUV[1], channelsYUV[2]))
    dst = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    dst = cv2.resize(dst, (1200, 900))

if choice == 3:
    # 漫画风
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 平滑
    img_blur = cv2.medianBlur(img_gray, 5)
    # 通过阈值提取轮廓
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize=9, C=3)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    # 颜色填充
    img_copy = im.copy()
    for _ in range(2):
        # 降低分辨率
        img_copy = cv2.pyrDown(img_copy)
    for _ in range(5):
        # 图像平滑，保留边缘
        img_copy = cv2.bilateralFilter(img_copy, d=9, sigmaColor=9, sigmaSpace=7)
    img_copy = cv2.resize(img_copy, (im.shape[1], im.shape[0]),
                          interpolation=cv2.INTER_CUBIC)
    # 与操作
    dst = cv2.bitwise_and(img_copy, img_edge)

import matplotlib.pyplot as plt

plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
cv2.waitKey(0)