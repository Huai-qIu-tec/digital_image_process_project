%%
% 实现彩色图像的平滑处理
% 采用手动实现padding
im = imread('D:\学习\数字图像处理\图片\CFig0.tif');
im = imnoise(im, 'salt & pepper',0.02);
global size_value;
size_value=inputdlg('请输入模板','模板',[1 40]);
size_value=str2double(char(size_value));
fp = padarray(im, [floor((size_value-1)/2) floor((size_value-1)/2)], 0, 'both');
h = fspecial('average',[size_value size_value]);
filterImg = imfilter(fp, h);
figure;
subplot(121);
imshow(im);title('加噪之后的原图');
subplot(122);
imshow(filterImg);title('平滑之后');

% 采用函数自带padding，重复填充
figure;
HSI = rgb2hsi(im);
H = HSI(:,:,1);
S = HSI(:,:,2);
I = HSI(:,:,3);
subplot(131)
imshow(im);title('RGB原图');
subplot(132)
imshow(HSI);title('HSI模式下的原图');
subplot(133)
im_filter_I = imfilter(I, h, 'replicate');
filterImg = hsi2rgb(cat(3, H, S, im_filter_I));
imshow(filterImg);title('在I通道平滑之后')

% 采用函数自带padding，镜像填充
figure;
im_filter = imfilter(im, h, 'symmetric');
subplot(121);
imshow(im);title('原图');
subplot(122);
imshow(im_filter);title('镜像填充');


% 非线性平滑 -- 中值滤波
[~, ~, c] = size(im);
if c == 3
    I1 = medfilt2(im(:, :, 1), [size_value size_value]);
    I2 = medfilt2(im(:, :, 2), [size_value size_value]);
    I3 = medfilt2(im(:, :, 3), [size_value size_value]);
    I = cat(3, I1, I2, I3);
    imshow(I);title('中值滤波');
else
    I = medfilt2(im, [size_value size_value]);
    imshow(I);
end
%%
% 实现彩色图像的锐化处理
im = imread('D:\学习\数字图像处理\图片\CFig0.tif');

% Laplace算子
[h, w, c] = size(im);
% 先对图片进行padding，再根据Laplace公式运用matlab的向量点乘运算计算每个卷积结果。
img_copy = im2double(im);
img_temp = padarray(img_copy, [1 1], 'replicate');  % padding重复填充
new_img = zeros(h, w, c);
pattern = -1 * ones(3,3);
pattern(2, 2) = 8;
[h1, w1, c] = size(img_temp);
for k = 1:c
    for i = 1:h1-2
        for j = 1:w1-2
            temp = img_temp(i:i+2, j:j+2, k);
            laplace_img = sum(sum(pattern .* temp));
            new_img(i, j, k) = laplace_img;
        end
    end
end
laplace_img = img_copy - new_img;
subplot(121)
imshow(im);title('原图');
subplot(1,2,2)
imshow(laplace_img);
title('3 × 3的Laplace算子');

% Sobel算子
figure;
subplot(221);
imshow(im);title('原图');
pattern_X = [-1 -2 -1;0 0 0 ;1 2 1];    % 水平方向的sobel算子
pattern_Y = pattern_X';                 % 垂直方向的sobel算子
im_before = im2double(im);
gradX = im_before - imfilter(im_before, pattern_X, 'replicate');
subplot(222);
imshow(gradX);title('水平方向Sobel锐化');
gradY = im_before - imfilter(im_before, pattern_Y, 'replicate');
subplot(223);
imshow(gradY);title('垂直方向Sobel锐化');
grad = sqrt(gradX.^2 + gradY.^2);
subplot(224);
imshow(grad);title('Sobel锐化');

sobel = fspecial('sobel');
im_sobel = im_before - imfilter(im_before, sobel);
imshow(im_sobel)

% Prewitt算子
figure;
subplot(221);
imshow(im);title('原图');
pattern_X = [-1 -1 -1;0 0 0 ;1 1 1];    % 水平方向的prewitt算子
pattern_Y = pattern_X';                 % 垂直方向的prewitt算子
im_before = im2double(im);
gradX = im_before - imfilter(im_before, pattern_X, 'replicate');
subplot(222);
imshow(gradX);title('水平方向Prewitt锐化');
gradY = im_before - imfilter(im_before, pattern_Y, 'replicate');
subplot(223);
imshow(gradY);title('垂直方向Prewitt锐化');
grad = sqrt(gradX.^2 + gradY.^2);
subplot(224);
imshow(grad);title('Prewitt锐化');
figure;
prewitt = fspecial('prewitt');
im_prewitt = im_before - imfilter(im_before, prewitt);
imshow(im_prewitt);

% roberts算子
im_copy = im;
[h, w, c] = size(im_copy);
im_before = im2double(im_copy);
im_after = im_before;
for k = 1:c
    for i = 1:h-1
        for j = 1:w-1
            im_after(i, j, k) = abs(im_before(i, j, k) - im_before(i+1, j+1, k)) + ...
                abs(im_before(i+1, j, k) - im_before(i, j+1, k));
        end
    end
end
im_Robert = im2double(im_copy) - im_after;
figure;
subplot(121)
imshow(im);title('原图');
subplot(122)
imshow(im_Robert);title('Roberts锐化图');

%%
% 实现彩色图像的边缘检测及分割处理
[VG, A, PPG] = colorgrad(im);
figure;
imshow(im2double(im) - VG);

mask=roipoly(im);
R = immultiply(mask,im(:,:,1));
G = immultiply(mask,im(:,:,2));
B = immultiply(mask,im(:,:,3));
g=cat(3,R,G,B);
figure,imshow(g);

[M,N,K]=size(g);
I=reshape(g,M*N,3);
idx=find(mask);
I=double(I(idx,1:3));
[C,m]=covmatrix(I);
d=diag(C);
sd=sqrt(d)';
E25=colorseg('euclidean',im,25,m);
figure,imshow(E25);

%%
% 学习ice函数的使用，通过阅读ice函数的脚本尝试自主开发曲线调节函数。
z = interp1q([0 255]', [0 255]', [0:255]');

ice('image', im)
