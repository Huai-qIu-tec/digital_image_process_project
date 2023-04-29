clc;clear;
img = imread('D:\学习\数字图像处理\图片\CFig0.tif');

whos

imshow(img);

info = imfinfo('D:\学习\数字图像处理\图片\CFig0.tif');

imwrite(img, 'newPic.jpg', 'quality', 10)
imshow('newPic.jpg')

imwrite(img, 'flower.bmp')
imshow('flower.bmp')


img2 = imread('D:\学习\数字图像处理\图片\CFig0.tif');
img3 = imread('D:\学习\数字图像处理\图片\Fig0.tif');

imfinfo('D:\学习\数字图像处理\图片\CFig0.tif')
imfinfo('D:\学习\数字图像处理\图片\Fig0.tif')

imshow(img2);
figure
imshow(img3);

img_bw = im2bw(img2);
imshow(img_bw)
