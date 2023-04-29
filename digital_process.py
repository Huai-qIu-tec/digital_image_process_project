import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r'D:\Picture\bear.jpg')
img_noise = img.copy()

# 显示图像
# cv2.imshow("Original", img)

row, col = img_noise.shape[0], img_noise.shape[1]

# 对图像加入噪声
for i in range(5000):
    x = np.random.randint(0, row)
    y = np.random.randint(0, col)
    img_noise[x, y, :] = 0

# 噪音图
source = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)

# 均值滤波
blur_mean_1 = cv2.blur(source, (3, 3))
blur_mean_2 = cv2.blur(source, (5, 5))
# 中值滤波
blur_median_1 = cv2.medianBlur(source, 3)
blur_median_2 = cv2.medianBlur(source, 5)
#高斯滤波
blur_gaussian_1 = cv2.GaussianBlur(source, (7, 7), 0)
blur_gaussian_2 = cv2.GaussianBlur(source, (9, 9), 0)

# cv2.imshow("Noise", img_noise)
# 均值滤波
plt.figure(figsize=(12, 3))
plt.subplot(141), plt.imshow(img[:, :, ::-1]), plt.title('original')
plt.subplot(142), plt.imshow(img_noise[:, :, ::-1]), plt.title('Noise')
plt.subplot(143), plt.imshow(blur_mean_1[:, :, ::-1]), plt.title('Mean Blur(3×3)')
plt.subplot(144), plt.imshow(blur_mean_2[:, :, ::-1]), plt.title('Mean Blur(5×5)')
plt.savefig(r'D:\学习\数字图像处理\Mean_Blur', dpi=600)
plt.show()


# 中值滤波
plt.figure(figsize=(12, 3))
plt.subplot(141), plt.imshow(img[:, :, ::-1]), plt.title('original')
plt.subplot(142), plt.imshow(img_noise[:, :, ::-1]), plt.title('Noise')
plt.subplot(143), plt.imshow(blur_median_1[:, :, ::-1]), plt.title('Median Blur(3×3)')
plt.subplot(144), plt.imshow(blur_median_1[:, :, ::-1]), plt.title('Median Blur(5×5)')
plt.savefig(r'D:\学习\数字图像处理\Median_Blur', dpi=600)
plt.show()


# 高斯滤波
plt.figure(figsize=(12, 3))
plt.subplot(141), plt.imshow(img[:, :, ::-1]), plt.title('original')
plt.subplot(142), plt.imshow(img_noise[:, :, ::-1]), plt.title('Noise')
plt.subplot(143), plt.imshow(blur_gaussian_1[:, :, ::-1]), plt.title('Gaussian Blur(5×5)')
plt.subplot(144), plt.imshow(blur_gaussian_2[:, :, ::-1]), plt.title('Gaussian Blur(7×7)')
plt.savefig(r'D:\学习\数字图像处理\Gaussian_Blur', dpi=600)
plt.show()



