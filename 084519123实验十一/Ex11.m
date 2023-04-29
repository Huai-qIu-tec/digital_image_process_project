%% 空域滤波 --> 频域滤波
% 平滑滤波器
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
subplot(131)
imshow(f);        title('原图')
h = fspecial('average', [5 5]);
AvgBlur = imfilter(f, h, 'replicate');
subplot(132)
imshow(AvgBlur);    title('平滑滤波器');
PQ = paddedsize(size(f));
H = freqz2(h, PQ(1), PQ(2));
H1 = ifftshift(H);
[m, n, c] = size(f);
if c == 1
    AvgFreqzBlur = dftfilt(f, H1);
else   
    AvgFreqzBlur1 = dftfilt(f(:, :, 1), H1);
    AvgFreqzBlur2 = dftfilt(f(:, :, 2), H1);
    AvgFreqzBlur3 = dftfilt(f(:, :, 3), H1);
    AvgFreqzBlur = cat(3, AvgFreqzBlur1, AvgFreqzBlur2, AvgFreqzBlur3);
end
subplot(133)
imshow(AvgFreqzBlur);    title('频域平滑滤波');
figure;
subplot(121)
imshow(abs(H), []);     title('频率域居中');
subplot(122)
imshow(abs(H1),[]);     title('频率域分散');

% 锐化
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
subplot(131)
imshow(f);        title('原图')
h = fspecial('sobel');
AvgBlur = imfilter(f, h, 'replicate');
subplot(132)
imshow(f - AvgBlur);    title('锐化滤波器');
PQ = paddedsize(size(f));
H = freqz2(h, PQ(1), PQ(2));
H1 = ifftshift(H);
AvgFreqzBlur = dftfilt(f, H);
subplot(133)
imshow(f - AvgFreqzBlur);    title('频域锐化滤波');

%%
% 低通滤波器
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
subplot(221)
imshow(f);  title('原图像');
I = im2double(f);
M = 2 * size(I, 1);
N = 2 * size(I, 2);
u = -M/2:(M/2-1);
v = -N/2:(N/2-1);
[U,V] = meshgrid(u, v);
D = sqrt(U.^2 + V.^2);
D0 = 80;
H1 = double(D <= D0);
J1=fftshift(fft2(I, size(H1, 1), size(H1, 2)));     
%时域图像转换为频域
K1=J1.*H1;
L1=ifft2(ifftshift(K1));           
%频域图像转换为时频
L1=L1(1:size(I,1), 1:size(I, 2));
subplot(222);imshow(L1);
title('理想低通滤波器');
n=6;
H2=1./(1+(D./D0).^(2*n));
J2=fftshift(fft2(I, size(H2, 1), size(H2, 2)));  %时域图像转换为频域
K2=J2.*H2;
L2=ifft2(ifftshift(K2));
%频域图像转换为时频
L2=L2(1:size(I,1), 1:size(I, 2));
subplot(223);imshow(L2);
title('巴特沃斯低通滤波器');
H3=exp(-(D.^2)./(2*(D0.^2)));
J3=fftshift(fft2(I, size(H3, 1), size(H3, 2)));  %时域图像转换为频域
K3=J3.*H3;
L3=ifft2(ifftshift(K3));
%频域图像转换为时频
L3=L3(1:size(I,1), 1:size(I, 2));
subplot(224);imshow(L3);
title('高斯低通滤波器');

%% 不同截至频率下的滤波器
% d0 = 10时的实用低通滤波器
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 10;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 1;
        else
            h = 0;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(221)
imshow(J1); title('d0 = 10的实用低通滤波器')
% d0 = 20时的实用低通滤波器
d0 = 20;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 1;
        else
            h = 0;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(222)
imshow(J1); title('d0 = 20的实用低通滤波器')
% d0 = 50时的实用低通滤波器
d0 = 50;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 1;
        else
            h = 0;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(223)
imshow(J1); title('d0 = 50的实用低通滤波器')

% d0 = 100时的实用低通滤波器
d0 = 100;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 1;
        else
            h = 0;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(224)
imshow(J1); title('d0 = 100的实用低通滤波器')

% d0 = 10巴特沃斯低通滤波
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 10;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d./d0)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(221)
imshow(J1); title('d0 = 10的巴特沃斯低通滤波器')

% d0 = 20巴特沃斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 20;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d./d0)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(222)
imshow(J1); title('d0 = 20的巴特沃斯低通滤波器')

% d0 = 50巴特沃斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 50;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d./d0)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(223)
imshow(J1); title('d0 = 50的巴特沃斯低通滤波器')

% d0 = 100巴特沃斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 100;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d./d0)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(224)
imshow(J1); title('d0 = 100的巴特沃斯低通滤波器')

% d0 = 10高斯低通滤波
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 10;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(221)
imshow(J1); title('d0 = 10的高斯低通滤波器')

% d0 = 20高斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 20;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(222)
imshow(J1); title('d0 = 20的高斯低通滤波器')

% d0 = 50高斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 50;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(223)
imshow(J1); title('d0 = 50的高斯低通滤波器')

% d0 = 100高斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 100;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(224)
imshow(J1); title('d0 = 100的高斯低通滤波器')

%% 高通滤波器
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
subplot(221)
imshow(f);  title('原图像');
I = im2double(f);
M = 2 * size(I, 1);
N = 2 * size(I, 2);
u = -M/2:(M/2-1);
v = -N/2:(N/2-1);
[U,V] = meshgrid(u, v);
D = sqrt(U.^2 + V.^2);
D0 = 80;
H1 = double(D >= D0);
J1 = fftshift(fft2(I, size(H1, 1), size(H1, 2)));     
%时域图像转换为频域
K1 = J1.*H1;
L1 = ifft2(ifftshift(K1));           
%频域图像转换为时频
L1 = L1(1:size(I,1), 1:size(I, 2));
subplot(222);imshow(im2double(f) - L1);
title('理想高通滤波器');
n=6;
H2 = 1./(1+(D0./D).^(2*n));
J2 = fftshift(fft2(I, size(H2, 1), size(H2, 2)));  %时域图像转换为频域
K2 = J2.*H2;
L2 = ifft2(ifftshift(K2));
%频域图像转换为时频
L2 = L2(1:size(I,1), 1:size(I, 2));
subplot(223);imshow(im2double(f) - L2);
title('巴特沃斯高通滤波器');
H3 = 1 - exp(-(D.^2)./(2*(D0.^2)));
J3 = fftshift(fft2(I, size(H3, 1), size(H3, 2)));  %时域图像转换为频域
K3 = J3.*H3;
L3 = ifft2(ifftshift(K3));
%频域图像转换为时频
L3 = L3(1:size(I,1), 1:size(I, 2));
subplot(224);imshow(im2double(f) - L3);
title('高斯高通滤波器');
%% 理想高通滤波器
% d0 = 10时的实用高通滤波器
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 10;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 0;
        else
            h = 1;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(221)
imshow(im2double(f) - J1); title('d0 = 10的实用高通滤波器')
% d0 = 20时的实用低通滤波器
d0 = 20;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 0;
        else
            h = 1;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(222)
imshow(im2double(f) - J1); title('d0 = 20的实用高通滤波器')
% d0 = 50时的实用低通滤波器
d0 = 50;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 0;
        else
            h = 1;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(223)
imshow(im2double(f) - J1); title('d0 = 50的实用高通滤波器')

% d0 = 100时的实用低通滤波器
d0 = 100;
h = zeros(m, n);
freq_img = zeros(m, n);
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d <= d0)
            h = 0;
        else
            h = 1;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(224)
imshow(im2double(f) - J1); title('d0 = 100的实用高通滤波器')
%% 巴特沃斯高通滤波器
% d0 = 10巴特沃斯低通滤波
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 10;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d0./d)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(221)
imshow(im2double(f) - J1); title('d0 = 10的巴特沃斯高通滤波器')

% d0 = 20巴特沃斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 20;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d0./d)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(222)
imshow(im2double(f) - J1); title('d0 = 20的巴特沃斯高通滤波器')

% d0 = 50巴特沃斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 50;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d0./d)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(223)
imshow(im2double(f) - J1); title('d0 = 50的巴特沃斯高通滤波器')

% d0 = 100巴特沃斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 100;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + d0./d)^(2 * N);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(224)
imshow(im2double(f) - J1); title('d0 = 100的巴特沃斯高通滤波器')
%% 高斯高通滤波器
% d0 = 10高斯低通滤波
f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 10;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1 - exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(221)
imshow(im2double(f) - J1); title('d0 = 10的高斯高通滤波器')

% d0 = 20高斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 20;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1 - exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(222)
imshow(im2double(f) - J1); title('d0 = 20的高斯高通滤波器')

% d0 = 50高斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 50;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1 - exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(223)
imshow(im2double(f) - J1); title('d0 = 50的高斯高通滤波器')

% d0 = 100高斯低通滤波

f = imread('D:\学习\数字图像处理\图片\Fig0.tif');
I = im2double(f);
F = fft2(I);
F_shift = ifftshift(F);
[m, n, c] = size(F_shift);
d0 = 100;
h = zeros(m, n);
freq_img = zeros(m, n);
N = 2;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1 - exp((-d.^N)./((d0.^N)));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(224)
imshow(im2double(f) - J1); title('d0 = 100的高斯高通滤波器')