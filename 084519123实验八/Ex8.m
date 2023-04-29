%%
%	利用imread( )函数读取一幅图像，用imshow(I,map)实现图像的显示，其中的map可以
%   使用matlab中任一预定的彩色映射；
[I, map] = imread('D:\学习\数字图像处理\图片\CFig9.tif');
subplot(221)
imshow(I);
title('无索引图像')
subplot(222)
imshow(I, map);
title('原图像');
subplot(223)
imshow(I, autumn);
title('autumn索引')
subplot(224)
imshow(I, winter);
title('winter索引')

% 彩色RGB图转ind
RGB = imread('D:\学习\数字图像处理\图片\bear.jpg');
[I, map] = rgb2ind(RGB, 256);
figure;
subplot(221)
imshow(I);
title('无索引图像')
subplot(222)
imshow(I, map);
title('原图像');
subplot(223)
imshow(I, autumn);
title('autumn索引')
subplot(224)
imshow(I, winter);
title('winter索引')

%%
% 2、利用rgb2gray命令实现彩色图像向灰度图像的转换操作；
RGB = imread('D:\学习\数字图像处理\图片\bear.jpg');
gray = rgb2gray(RGB);
imshow(gray);

%%
%3、利用rgb2ind命令实现彩色图像向索引图像的转换操作；
RGB = imread('D:\学习\数字图像处理\图片\bear.jpg');
[I, map] = rgb2ind(RGB, 256);
imshow(I, map);
%%
%4、利用im2bw命令实现彩色图像\索引图像\灰度图像向二值图像的转换操作；
RGB = imread('D:\学习\数字图像处理\图片\bear.jpg');
GRAY = imread('D:\学习\数字图像处理\图片\Fig2.tif');
[IND, map] = imread('D:\学习\数字图像处理\图片\CFig9.tif');
RGB_BW = im2bw(RGB);
GRAY_BW = im2bw(GRAY);
IND_BW = im2bw(IND, map);
figure;
subplot(131)
imshow(RGB_BW);
title('彩色2二值');
subplot(132)
imshow(GRAY_BW);
title('灰度2二值');
subplot(133)
imshow(IND_BW);
title('索引2二值');
%%
%5、利用ind2rgb命令实现索引图像向RGB图像的转换操作。
[IND, map] = imread('D:\学习\数字图像处理\图片\CFig9.tif');
RGB = ind2rgb(IND, jet);
imshow(RGB);
title('ind2rgb');

%%
%6、利用ind2gray命令实现索引图像向灰度图像的转换操作。
[IND, map] = imread('D:\学习\数字图像处理\图片\CFig9.tif');
GRAY = ind2gray(IND, summer);
subplot(121)
imshow(GRAY);
title('ind2rgb[map=summer]');
subplot(122)
GRAY = ind2gray(IND, jet);
imshow(GRAY);
title('ind2rgb[map=jet]');
%%
%7、利用gray2ind, grayslice命令实现灰度图像向索引图像的转换操作。
gray_image=imread('cameraman.tif');
subplot(2,2,1),imshow(gray_image);
title('原图');
[X,map]=gray2ind(gray_image,8);
title('颜色数为8');
subplot(2,2,2),imshow(X,map);
[X,map]=gray2ind(gray_image,64);
title('颜色数为64');
subplot(2,2,3),imshow(X,map);
[X,map]=gray2ind(gray_image,256);
subplot(2,2,4),imshow(X,map);
title('颜色数为256');

gray_image=imread('cameraman.tif');
subplot(2,2,1),imshow(gray_image);
title('原图')
X=grayslice(gray_image,16);
subplot(2,2,2),imshow(X,summer(16));
title('n=16时summber图')
X=grayslice(gray_image,32);
subplot(2,2,3),imshow(X,summer(32));
title('n=32时summer图')
X=grayslice(gray_image,64);
subplot(2,2,4),imshow(X,summer(64));
title('n=64时summer图')
%%
%8、实现图像抖动效果的添加
RGB = imread('D:\学习\数字图像处理\图片\bear.jpg');
[X,map]=rgb2ind(RGB, 8,'nodither');
subplot(1,2,1),imshow(X,map);
title('未抖动的索引图像');
[X1,map1] = rgb2ind(RGB,8,'dither');
subplot(1,2,2),imshow(X1,map1)
title('抖动处理的彩色索引图像')
%%
%9、查看不同颜色模型各个分量显示效果

%RGB
RGB = imread('D:\学习\数字图像处理\图片\bear.jpg');
subplot(221)
imshow(RGB)
title('原图')
subplot(222)
imshow(cat(3, RGB(:,:,1), zeros(size(RGB(:,:,2))), zeros(size(RGB(:,:,3)))))
title('R分量')
subplot(223)
imshow(cat(3, zeros(size(RGB(:,:,1))), RGB(:,:,2), zeros(size(RGB(:,:,3)))))
title('G分量')
subplot(224)
imshow(cat(3, zeros(size(RGB(:,:,1))), zeros(size(RGB(:,:,2))), RGB(:,:,3)))
title('B分量')

%HSV
HSV = rgb2hsv(RGB);
HSV_H = HSV(:, :, 1);
HSV_S = HSV(:, :, 2);
HSV_V = HSV(:, :, 3);
subplot(221)
imshow(HSV)
title('原图')
subplot(222)
imshow(cat(3, HSV_H, zeros(size(HSV_S)), zeros(size(HSV_V))))
title('色度分量')
subplot(223)
imshow(cat(3, zeros(size(HSV_H)), HSV_S, zeros(size(HSV_V))))
title('饱和度分量')
subplot(224)
imshow(cat(3, zeros(size(HSV_H)), zeros(size(HSV_S)), HSV_V))
title('明度分量')

%CMY
RGB = imread('D:\学习\数字图像处理\图片\bear.jpg');
CMY = imcomplement(RGB);
RGB = im2double(RGB);
R = RGB(:, :, 1);
G = RGB(:, :, 2);
B = RGB(:, :, 3);
[m, n, ~] = size(RGB);
subplot(221)
imshow(CMY);
title('原图')
subplot(222)
C = ones(m, n) - R;
M = ones(m, n) - G;
Y = ones(m, n) - B;
imshow(cat(3, C, zeros(size(M)), zeros(size(Y))))
title('青色分量')
subplot(223)

imshow(cat(3, zeros(size(C)), M, zeros(size(Y))))
title('品红色分量')
subplot(224)

imshow(cat(3, zeros(size(C)), zeros(size(Y)), Y))
title('黄色分量')

%NTSC
RGB = imread('D:\学习\数字图像处理\图片\CFig8.png');
NTSC = rgb2ntsc(RGB);
Y = NTSC(:, :, 1);
I = NTSC(:, :, 2);
Q = NTSC(:, :, 3);
subplot(221)
imshow(NTSC)
title('原图')
subplot(222)
imshow(cat(3, Y, zeros(size(I)), zeros(size(Q))))
title('亮度分量')
subplot(223)
imshow(cat(3, zeros(size(Y)), I, zeros(size(Q))))
title('色调分量')
subplot(224)
imshow(cat(3, zeros(size(Y)), zeros(size(I)), Q))
title('饱和度')

%YUV
RGB = imread('D:\学习\数字图像处理\图片\CFig8.png');
RGB = im2double(RGB);
R = RGB(:, :, 1);
G = RGB(:, :, 2);
B = RGB(:, :, 3);
YUV = rgb2ycbcr(RGB);
Y = YUV(:, :, 1);
U = YUV(:, :, 2);
V = YUV(:, :, 3);
subplot(221)
imshow(NTSC)
title('原图')
subplot(222)
imshow(cat(3, Y, zeros(size(U)), zeros(size(V))))
title('亮度分量')
subplot(223)
U = R - im2double(Y);
imshow(cat(3, zeros(size(Y)), U, zeros(size(V))))
title('R－Y分量')
subplot(224)
V = B - Y;
imshow(cat(3, zeros(size(U)), zeros(size(I)), V))
title('B－Y分量')