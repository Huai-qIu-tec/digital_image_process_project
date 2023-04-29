%%
%	����imread( )������ȡһ��ͼ����imshow(I,map)ʵ��ͼ�����ʾ�����е�map����
%   ʹ��matlab����һԤ���Ĳ�ɫӳ�䣻
[I, map] = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig9.tif');
subplot(221)
imshow(I);
title('������ͼ��')
subplot(222)
imshow(I, map);
title('ԭͼ��');
subplot(223)
imshow(I, autumn);
title('autumn����')
subplot(224)
imshow(I, winter);
title('winter����')

% ��ɫRGBͼתind
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
[I, map] = rgb2ind(RGB, 256);
figure;
subplot(221)
imshow(I);
title('������ͼ��')
subplot(222)
imshow(I, map);
title('ԭͼ��');
subplot(223)
imshow(I, autumn);
title('autumn����')
subplot(224)
imshow(I, winter);
title('winter����')

%%
% 2������rgb2gray����ʵ�ֲ�ɫͼ����Ҷ�ͼ���ת��������
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
gray = rgb2gray(RGB);
imshow(gray);

%%
%3������rgb2ind����ʵ�ֲ�ɫͼ��������ͼ���ת��������
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
[I, map] = rgb2ind(RGB, 256);
imshow(I, map);
%%
%4������im2bw����ʵ�ֲ�ɫͼ��\����ͼ��\�Ҷ�ͼ�����ֵͼ���ת��������
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
GRAY = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig2.tif');
[IND, map] = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig9.tif');
RGB_BW = im2bw(RGB);
GRAY_BW = im2bw(GRAY);
IND_BW = im2bw(IND, map);
figure;
subplot(131)
imshow(RGB_BW);
title('��ɫ2��ֵ');
subplot(132)
imshow(GRAY_BW);
title('�Ҷ�2��ֵ');
subplot(133)
imshow(IND_BW);
title('����2��ֵ');
%%
%5������ind2rgb����ʵ������ͼ����RGBͼ���ת��������
[IND, map] = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig9.tif');
RGB = ind2rgb(IND, jet);
imshow(RGB);
title('ind2rgb');

%%
%6������ind2gray����ʵ������ͼ����Ҷ�ͼ���ת��������
[IND, map] = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig9.tif');
GRAY = ind2gray(IND, summer);
subplot(121)
imshow(GRAY);
title('ind2rgb[map=summer]');
subplot(122)
GRAY = ind2gray(IND, jet);
imshow(GRAY);
title('ind2rgb[map=jet]');
%%
%7������gray2ind, grayslice����ʵ�ֻҶ�ͼ��������ͼ���ת��������
gray_image=imread('cameraman.tif');
subplot(2,2,1),imshow(gray_image);
title('ԭͼ');
[X,map]=gray2ind(gray_image,8);
title('��ɫ��Ϊ8');
subplot(2,2,2),imshow(X,map);
[X,map]=gray2ind(gray_image,64);
title('��ɫ��Ϊ64');
subplot(2,2,3),imshow(X,map);
[X,map]=gray2ind(gray_image,256);
subplot(2,2,4),imshow(X,map);
title('��ɫ��Ϊ256');

gray_image=imread('cameraman.tif');
subplot(2,2,1),imshow(gray_image);
title('ԭͼ')
X=grayslice(gray_image,16);
subplot(2,2,2),imshow(X,summer(16));
title('n=16ʱsummberͼ')
X=grayslice(gray_image,32);
subplot(2,2,3),imshow(X,summer(32));
title('n=32ʱsummerͼ')
X=grayslice(gray_image,64);
subplot(2,2,4),imshow(X,summer(64));
title('n=64ʱsummerͼ')
%%
%8��ʵ��ͼ�񶶶�Ч�������
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
[X,map]=rgb2ind(RGB, 8,'nodither');
subplot(1,2,1),imshow(X,map);
title('δ����������ͼ��');
[X1,map1] = rgb2ind(RGB,8,'dither');
subplot(1,2,2),imshow(X1,map1)
title('��������Ĳ�ɫ����ͼ��')
%%
%9���鿴��ͬ��ɫģ�͸���������ʾЧ��

%RGB
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
subplot(221)
imshow(RGB)
title('ԭͼ')
subplot(222)
imshow(cat(3, RGB(:,:,1), zeros(size(RGB(:,:,2))), zeros(size(RGB(:,:,3)))))
title('R����')
subplot(223)
imshow(cat(3, zeros(size(RGB(:,:,1))), RGB(:,:,2), zeros(size(RGB(:,:,3)))))
title('G����')
subplot(224)
imshow(cat(3, zeros(size(RGB(:,:,1))), zeros(size(RGB(:,:,2))), RGB(:,:,3)))
title('B����')

%HSV
HSV = rgb2hsv(RGB);
HSV_H = HSV(:, :, 1);
HSV_S = HSV(:, :, 2);
HSV_V = HSV(:, :, 3);
subplot(221)
imshow(HSV)
title('ԭͼ')
subplot(222)
imshow(cat(3, HSV_H, zeros(size(HSV_S)), zeros(size(HSV_V))))
title('ɫ�ȷ���')
subplot(223)
imshow(cat(3, zeros(size(HSV_H)), HSV_S, zeros(size(HSV_V))))
title('���Ͷȷ���')
subplot(224)
imshow(cat(3, zeros(size(HSV_H)), zeros(size(HSV_S)), HSV_V))
title('���ȷ���')

%CMY
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
CMY = imcomplement(RGB);
RGB = im2double(RGB);
R = RGB(:, :, 1);
G = RGB(:, :, 2);
B = RGB(:, :, 3);
[m, n, ~] = size(RGB);
subplot(221)
imshow(CMY);
title('ԭͼ')
subplot(222)
C = ones(m, n) - R;
M = ones(m, n) - G;
Y = ones(m, n) - B;
imshow(cat(3, C, zeros(size(M)), zeros(size(Y))))
title('��ɫ����')
subplot(223)

imshow(cat(3, zeros(size(C)), M, zeros(size(Y))))
title('Ʒ��ɫ����')
subplot(224)

imshow(cat(3, zeros(size(C)), zeros(size(Y)), Y))
title('��ɫ����')

%NTSC
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig8.png');
NTSC = rgb2ntsc(RGB);
Y = NTSC(:, :, 1);
I = NTSC(:, :, 2);
Q = NTSC(:, :, 3);
subplot(221)
imshow(NTSC)
title('ԭͼ')
subplot(222)
imshow(cat(3, Y, zeros(size(I)), zeros(size(Q))))
title('���ȷ���')
subplot(223)
imshow(cat(3, zeros(size(Y)), I, zeros(size(Q))))
title('ɫ������')
subplot(224)
imshow(cat(3, zeros(size(Y)), zeros(size(I)), Q))
title('���Ͷ�')

%YUV
RGB = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig8.png');
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
title('ԭͼ')
subplot(222)
imshow(cat(3, Y, zeros(size(U)), zeros(size(V))))
title('���ȷ���')
subplot(223)
U = R - im2double(Y);
imshow(cat(3, zeros(size(Y)), U, zeros(size(V))))
title('R��Y����')
subplot(224)
V = B - Y;
imshow(cat(3, zeros(size(U)), zeros(size(I)), V))
title('B��Y����')