""" 从视频读取帧保存为图片"""
'''
1.由BGR空间转到YCrCb空间，
    在RGB空间里人脸的肤色受亮度影响相当大，所以肤色点很难从非肤色点中分离出来，也就是说在此空间经过处理后，肤色点是离散的点，中间嵌有很多非肤色，
    如果把RGB转为YCrCb空间的话，可以忽略Y(亮度)的影响，因为该空间受亮度影响很小，肤色会产生很好的类聚。
    这样就把三维的空间将为二维的CrCb，肤色点会形成一定得形状，如：人脸的话会看到一个人脸的区域，手臂的话会看到一条手臂的形态。
    
    在这部分种需要对Cr空间内进行高斯算子进行锐化,再用OTSU阈值处理cv2.threshold和掩膜与计算
    OTSU阈值处理：
    https://blog.csdn.net/weixin_43414877/article/details/116699343?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.highlightwordscore&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.highlightwordscore
2.轮廓处理
    轮廓处理需要在灰度图像上进行分割,因此需要把BGR转成GRAY 
    轮廓处理的话主要用到两个函数，
    cv2.findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
    image:寻找轮廓的图像
    mode:轮廓的检索模式 (cv2.RETR_EXTERNAL:表示只检测外轮廓)
    method:轮廓的近似办法 (cv2.CHAIN_APPROX_NONE:存储所有的轮廓点,相邻的两个点的像素位置差不超过1，即max(abs(x1-x2)，abs（y2-y1))==1)
    
    和
    
    cv2.drawContours(image, contours, contourIdx, color+[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]]) 
    image:指明在哪幅图像上绘制轮廓；
    contours:轮廓本身，在Python中是一个list。
    contourIdx:指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。
    thickness:表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。
    
    这部分主要的问题是提取到的轮廓有很多个，但是我们只需要手的轮廓，所以我们要用sorted函数找到最大的轮廓。
    
'''

import cv2
import numpy as np

# cap = cv2.VideoCapture(0)#读取文件
cap = cv2.VideoCapture(0)  # 读取摄像头


# 皮肤检测
def A(img):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
    res = cv2.bitwise_and(img, img, mask=skin)
    return res


def B(img):
    # binaryimg = cv2.Canny(Laplacian, 50, 200) #二值化，canny检测
    h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
    contour = h[0]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)  # 对轮廓区域面积进行排序
    # contourmax = contour[0][:, 0, :]#保留区域面积最大的轮廓点坐标
    bg = np.ones(dst.shape, np.uint8) * 255  # 创建白色幕布
    ret = cv2.drawContours(bg, contour[0], -1, (0, 0, 0), 3)  # 绘制黑色轮廓
    return ret


while (True):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    src = cv2.resize(frame, (400, 350), interpolation=cv2.INTER_CUBIC)  # 窗口大小
    cv2.rectangle(src, (90, 60), (300, 300), (0, 255, 0))  # 框出截取位置
    roi = src[60:300, 90:300]  # 获取手势框图

    res = A(roi)  # 进行肤色检测
    cv2.imshow("0", roi)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)

    contour = B(Laplacian)  # 轮廓处理
    cv2.imshow("2", contour)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
