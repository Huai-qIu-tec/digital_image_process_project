%%
% 编程实现二维图像的离散傅立叶变换及傅立叶反变换
clear
clc
% 数据量太大，定义式算不出来
im = imread('D:\学习\数字图像处理\图片\Fig2.tif');
A=[12 13 19 13 12;14 16 25 16 14;24 31 99 31 24;14 16 25 16 14;12 13 19 13 12];
[m, n, c] = size(A);
Fouier_img = zeros(m, n);
for u = 1:m
    for v = 1:n
        for x = 1:m
            for y = 1:m
                Fouier_img(u, v) = Fouier_img(u, v) + A(x,y) * exp(-1i * 2 * pi * (u * x/ m + v * y/ n));
            end
        end
    end
end
subplot(121)
image(real(Fouier_img))
colormap(gray)
title('公式法实现fft')

subplot(122)
F = fft2(A);
image(real(F))
colormap(gray)
title('matlab自带fft')
% 矩阵计算
im = imread('D:\学习\数字图像处理\图片\bear.jpg');
[m, n, c] = size(im);
Gm = zeros(m);
Gn = zeros(n);

% 初始化Gm
for i = 1:m
    for j = 1:m
        Gm(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / m);
    end
end
% 初始化Gn
for i = 1:n
    for j = 1:n
        Gn(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / n);
    end
end
% 对图片的三个通道分别做傅里叶变换
Fouier_img_mt_1 = Gm * double(im(:, :, 1)) * Gn;
Fouier_img_mt_2 = Gm * double(im(:, :, 2)) * Gn;
Fouier_img_mt_3 = Gm * double(im(:, :, 3)) * Gn;
Fouier_img_mt = cat(3, Fouier_img_mt_1, Fouier_img_mt_2, Fouier_img_mt_3);
subplot(121)
imshow(Fouier_img_mt)
title('向量法实现fft')

% matlab自带的fft2
subplot(122)
F = fft2(im);
imshow(F);
title('matlab自带fft')

Real = abs(Fouier_img_mt);
Normalization_Real = (Real - min(min(Real))) ./ (max(max(Real)) - min(min(Real))) * 255;
imshow(Normalization_Real)

% iFFt
% 求矩阵的逆
G3 = inv(Gm);
G4 = inv(Gn);
% 对图片的三个通道分别做傅里叶反变换
iFouier_img_1 = G3 * Fouier_img_mt_1 * G4 / 255;
iFouier_img_2 = G3 * Fouier_img_mt_2 * G4 / 255;
iFouier_img_3 = G3 * Fouier_img_mt_3 * G4 / 255;
iFouier_img_mt = cat(3, iFouier_img_1, iFouier_img_2, iFouier_img_3);
imshow(iFouier_img_mt)
title('傅里叶反变换');
%%
% 利用fft2 ()函数实现一幅图像的傅里叶正变换，并使用subplot（）函数及subimage（）
% 函数实现原图像及频谱图的对比显示；
F = fft2(im);
subplot(221); imshow(im); title('原图像')
subplot(222); imshow(F); title('傅里叶变换频谱图')

%%
% 利用ifft2 ()函数实现频谱图像的傅里叶反变换，并使用subplot（）函数及subimage（）
% 函数实现频谱图及原图像的对比显示；
f = ifft2(F);
k = ifft2(f) / 255;
subplot(223); imshow(im); title('傅里叶反变换')
subplot(224); imshow(k); title('傅里叶反变换频谱图')

%%
% 对比自定义实现的傅立叶变换函数与Matlab函数fft2 ()对同一图像转换的结果以及执行时间
im = imread('D:\学习\数字图像处理\图片\bear.jpg');
t1 = cputime;
[m, n, c] = size(im);
Gm = zeros(m);
Gn = zeros(n);

% 初始化Gm
for i = 1:m
    for j = 1:m
        Gm(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / m);
    end
end
% 初始化Gn
for i = 1:n
    for j = 1:n
        Gn(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / n);
    end
end
% 对图片的三个通道分别做傅里叶变换
Fouier_img_mt_1 = Gm * double(im(:, :, 1)) * Gn;
Fouier_img_mt_2 = Gm * double(im(:, :, 2)) * Gn;
Fouier_img_mt_3 = Gm * double(im(:, :, 3)) * Gn;
Fouier_img_mt = cat(3, Fouier_img_mt_1, Fouier_img_mt_2, Fouier_img_mt_3);
t2 = cputime;
t = t2 - t1;
% 自定义傅里叶变换时间
t 
% 自带傅里叶变换时间
t1 = cputime;
F = fft2(im);
t2 = cputime;
t = t2 - t1;
t

%%
% 对比自定义实现的傅立叶反变换函数与Matlab函数ifft2 ()对同一图像转换的结果以及执行时间。
im = imread('D:\学习\数字图像处理\图片\bear.jpg');
t1 = cputime;
[m, n, c] = size(im);
Gm = zeros(m);
Gn = zeros(n);
% 初始化Gm
for i = 1:m
    for j = 1:m
        Gm(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / m);
    end
end
% 初始化Gn
for i = 1:n
    for j = 1:n
        Gn(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / n);
    end
end
% iFFt
% 求矩阵的逆
G3 = inv(Gm);
G4 = inv(Gn);
% 对图片的三个通道分别做傅里叶反变换
iFouier_img_1 = G3 * Fouier_img_mt_1 * G4 / 255;
iFouier_img_2 = G3 * Fouier_img_mt_2 * G4 / 255;
iFouier_img_3 = G3 * Fouier_img_mt_3 * G4 / 255;
iFouier_img_mt = cat(3, iFouier_img_1, iFouier_img_2, iFouier_img_3);
t2 = cputime;
t = t2 - t1;
% 自定义反变换时间
t

% matlab反变换时间
t1 = cputime;
f = ifft2(F);
t2 = cputime;
t = t2 - t1;
t

%%
% 掌握fftshift()、ifftshift()等函数的使用。
im = imread('D:\学习\数字图像处理\图片\Fig2.tif');
subplot(221)
imshow(im);title('原图')
F = fft2(im);
subplot(222)
imshow(F); title('傅里叶变换频谱图')

% fftshift
Shift_F = fftshift(F);
imshow(Shift_F)
A = abs(Shift_F);
Normalization_F = (A - min(min(A))) ./ (max(max(A)) - min(min(A))) * 255;
subplot(223)
imshow(Normalization_F)
title('原点居中')

% ifftshift
i_Shift_F = ifftshift(Shift_F);
AI = abs(i_Shift_F);
Normalization_iF = (AI - min(min(AI))) ./ (max(max(AI)) - min(min(AI))) * 255;
subplot(224)
imshow(Normalization_iF)
title('原点四周分散')

%%
% 掌握傅立叶频谱得到幅度谱和相位谱的处理方法。
im = imread('D:\学习\数字图像处理\图片\Fig4.bmp');
F = fft2(im);
Shift_F = fftshift(F);
sF = log(abs(Shift_F));   %获得傅里叶变换的幅度谱
phF = log(angle(Shift_F) * 180 / pi);   %获得傅里叶变换的相位谱
subplot(121);
imshow(sF,[]); %显示图像的幅度谱，参数与[]是为了将sA的值线形拉伸
title('傅里叶变换幅度谱');
subplot(122);
imshow(phF,[]); %显示图像傅里叶变换的相位谱
title('傅里叶变换的相位谱');

%%
% 对不同频率的正弦光栅化图像进行频域转换，对比正弦光栅化图像与幅度谱及相位谱之间的属性关系。
% 首先产生正弦光栅条纹
subplot(331)
I = zeros(512, 512);
for i = 1:512
    for j = 1:512
        I(i, j) = 127 + 126*cos(2 * pi * j/ 128);
    end
end
I_low = mat2gray(I);
I_low = I_low(:,:,1);
imshow(I_low); title('低频图像')

subplot(332)
I = zeros(512, 512);
for i = 1:512
    for j = 1:512
        I(i, j) = 127 + 126*cos(2 * pi * j/ 64);
    end
end
I_middle = mat2gray(I);
I_middle = I_middle(:,:,1);
imshow(I_middle); title('中频图像')

subplot(333)
I = zeros(512, 512);
for i = 1:512
    for j = 1:512
        I(i, j) = 127 + 126*cos(2 * pi * j/ 16);
    end
end
I_high = mat2gray(I);
I_high = I_high(:,:,1);
imshow(I_high); title('高频图像')

low_F = fft2(I_low);
middle_F = fft2(I_middle);
high_F = fft2(I_high);

Shift_low = fftshift(low_F);
Shift_middle = fftshift(middle_F);
Shift_high = fftshift(high_F);

s_low = log(abs(Shift_low));
s_middle = log(abs(Shift_middle));
s_high = log(abs(Shift_high));

ph_low = log(angle(Shift_low)*180/pi);
ph_middle = log(angle(Shift_middle)*180/pi);
ph_high = log(angle(Shift_high)*180/pi);

subplot(334);imshow(s_low,[]); title('低频图像幅度谱');
subplot(337);imshow(ph_low,[]);title('低频图像相位谱');
subplot(335);imshow(s_middle,[]);title('中频图像幅度谱');
subplot(338);imshow(ph_middle,[]);title('中频图像相位谱');
subplot(336);imshow(s_high,[]);title('高频图像幅度谱');
subplot(339);imshow(ph_high,[]);title('高频图像相位谱');