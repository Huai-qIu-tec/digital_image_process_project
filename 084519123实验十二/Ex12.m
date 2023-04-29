%% 1.模拟各种噪声
% 高斯噪声
Gaussian_noise = imnoise2('gaussian', 10000, 1, 0.1);
subplot(241)
histogram(Gaussian_noise, 100);
title('高斯噪声');

% 均匀噪声
Uniform_noise = imnoise2('uniform', 10000, 1, 0.1);
subplot(242)
histogram(Uniform_noise, 100);
title('均匀噪声');

% 对数正态噪声
Lognormal_noise = imnoise2('lognormal', 10000, 1, 1, 0.25);
subplot(243)
histogram(Lognormal_noise, 100);
title('对数正态噪声');

% 瑞利噪声
Rayleigh_noise = imnoise2('rayleigh', 10000, 1, 0.1);
subplot(244)
histogram(Rayleigh_noise, 100);
title('瑞利噪声');

% 指数噪声
exp_noise = imnoise2('exponential', 10000, 1, 1);
subplot(245)
histogram(exp_noise, 100);
title('指数噪声');

% 爱尔兰噪声
erlang_noise = imnoise2('erlang', 10000, 1, 2, 5);
subplot(246)
histogram(erlang_noise, 100);
title('爱尔兰噪声');

% 椒盐噪声
SaltPepper_noise = imnoise2('salt & pepper', 10000, 1, 0.05, 0.05);
subplot(247)
histogram(SaltPepper_noise, 100);
title('椒盐噪声');

% 周期噪声
C = [0 64;0 128;32 32;64 0;128 0;-32 32];
[r, R, S] = imnoise3(512,512,C);
subplot(2,3,1),imshow(S,[]);title('频谱') %显示频谱
subplot(2,3,4),imshow(r,[]);title('正弦噪声') %显示空间正弦噪声模式
C = [0 32;0 64;16 16;32 0;64 0;-16 16]; %改变冲击位置，观察频谱和空间正弦噪声模式变化 [r,R,S]=imnoise3(512,512,C);
subplot(2,3,2),imshow(S,[]);title('频谱')
subplot(2,3,5),imshow(r,[]);title('正弦噪声')
C = [6 32;-2 2]; %改变冲击位置，观察空间正弦噪声模式变化
[r, R, S] = imnoise3(512,512,C);
subplot(2,3,3),imshow(r,[]);title('改变冲击位置正弦噪声')
A = [1 5]; %使用非默认振幅向量A,观察空间正弦噪声模式变化
[r, R, S] = imnoise3(512,512,C,A);
subplot(2,3,6),imshow(r,[]);title('非默认振幅向量正弦噪声')
%% 2.添加噪声和模糊化处理
im = imread('D:\学习\数字图像处理\图片\Fig0.tif');
[m, n, c] = size(im);
% 利用椒盐噪声污染图片
im_saltpepper = im;
R = imnoise2('salt & pepper', m, n, 0, 0.1); % 盐粒污染图像
index_salt = find(R == 1);
im_saltpepper(index_salt) = 255;

R = imnoise2('salt & pepper', m, n, 0.1, 0); % 盐粒污染图像
index_pepper = find(R == 0);
im_saltpepper(index_pepper) = 0;
subplot(231)
imshow(im_saltpepper);   title('椒盐噪声图');
h = fspecial('average', [3 3]);
im_saltpepper = imfilter(im_saltpepper, h, 'replicate'); 
subplot(234)
imshow(im_saltpepper);   title('椒盐噪声平滑图')

% 添加高斯噪声
R_gaussian = imnoise2('gaussian', m, n, 0, 0.1);
im_gaussian = im2double(im) + R_gaussian;
subplot(232)
imshow(im_gaussian);    title('高斯噪声图')
h = fspecial('average', [3 3]);
im_gaussian = imfilter(im_gaussian, h, 'replicate'); 
subplot(235)
imshow(im_gaussian);    title('高斯噪声平滑图')

% 添加均匀噪声
R_uniform = imnoise2('uniform', m, n, -0.2, 0.2);
im_uniform = im2double(im) + R_uniform;
subplot(233)
imshow(im_uniform);     title('均匀噪声图')
h = fspecial('average', [3 3]);
im_uniform = imfilter(im_uniform, h, 'replicate'); 
subplot(236)
imshow(im_uniform);    title('均匀噪声平滑图')

%% 验证算术均值、几何均值、调和均值、逆调和均值等空间域滤波法
% 图片加噪
im = imread('D:\学习\数字图像处理\图片\Fig0.tif');
R_gaussian = imnoise(im, 'gaussian');
R_saltpepper = imnoise(im, 'Salt & Pepper');
R_poisson = imnoise(im, 'poisson');

% 算术均值
subplot(231);   imshow(R_gaussian);     title('高斯噪声');
h = fspecial('average', [3 3]);
mean_filter = imfilter(R_gaussian, h, 'replicate');
subplot(234);   imshow(mean_filter);     title('高斯算术均值平滑图');

subplot(232);   imshow(R_saltpepper);     title('高斯噪声');
mean_filter = imfilter(R_gaussian, h, 'replicate');
subplot(235);   imshow(mean_filter);     title('椒盐算术均值平滑图');

subplot(233);   imshow(R_poisson);     title('高斯噪声');
mean_filter = imfilter(R_gaussian, h, 'replicate');
subplot(236);   imshow(mean_filter);     title('泊松算术均值平滑图');

% 几何均值
% 高斯噪声

subplot(231);    imshow(R_gaussian);     title('高斯噪声图');
padding_img = padarray(R_gaussian, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = prod(prod(padding_img(i:i+2, j:j+2)))^(1/9);
    end
end
subplot(234);   imshow(new_img(1:256, 1:256));  title('高斯几何均值滤波');

% 泊松噪声

subplot(232);    imshow(R_poisson);     title('高斯噪声图');
padding_img = padarray(R_poisson, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = prod(prod(padding_img(i:i+2, j:j+2)))^(1/9);
    end
end
subplot(235);   imshow(new_img(1:256, 1:256));  title('泊松几何均值滤波');

% 椒盐噪声
subplot(233);    imshow(R_saltpepper);     title('高斯噪声图');
padding_img = padarray(R_saltpepper, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = prod(prod(padding_img(i:i+2, j:j+2)))^(1/9);
    end
end
subplot(236);   imshow(new_img(1:256, 1:256));  title('椒盐几何均值滤波');

% 调和均值 对于“盐”噪声效果好，但不适用于“胡椒”噪声
% 高斯噪声
subplot(231);   imshow(R_gaussian);  title('高斯噪声');
padding_img = padarray(R_gaussian, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = 9 / sum(sum(1./(padding_img(i:i+2, j:j+2))));
    end
end
subplot(234)
imshow(new_img(1:256, 1:256));
title('调和平均高斯噪声');

% 椒盐噪声
subplot(232);   imshow(R_saltpepper);  title('椒盐噪声');
padding_img = padarray(R_saltpepper, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = 9 / sum(sum(1./(padding_img(i:i+2, j:j+2))));
    end
end
subplot(235)
imshow(new_img(1:256, 1:256))
title('调和平均椒盐噪声');

% 泊松噪声
subplot(233);   imshow(R_poisson);  title('泊松噪声');
padding_img = padarray(R_poisson, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = 9 / sum(sum(1./(padding_img(i:i+2, j:j+2))));
    end
end
subplot(236)
imshow(new_img(1:256, 1:256))
title('调和平均泊松噪声');

% 逆调和均值滤波器 
Q = -1;
R_saltpepper1 = R_saltpepper;
subplot(231);   imshow(R_saltpepper1);  title('椒盐噪声');
padding_img = padarray(R_saltpepper1, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = sum(sum(padding_img(i:i+2, j:j+2) .^ (Q + 1))) / sum(sum(padding_img(i:i+2, j:j+2) .^ Q));
    end
end
subplot(234)
imshow(new_img(1:256, 1:256))
title('Q为负值时的逆调和均值算法')

Q = 1;
R_saltpepper2 = R_saltpepper;
subplot(232);   imshow(R_saltpepper2);  title('椒盐噪声');
padding_img = padarray(R_saltpepper2, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = sum(sum(padding_img(i:i+2, j:j+2) .^ (Q + 1))) / sum(sum(padding_img(i:i+2, j:j+2) .^ Q));
    end
end
subplot(235)
imshow(new_img(1:256, 1:256))
title('Q为正值时的逆调和均值算法')

Q = 0;
R_saltpepper3 = R_saltpepper;
subplot(233);   imshow(R_saltpepper3);  title('椒盐噪声');
padding_img = padarray(R_saltpepper3, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = sum(sum(padding_img(i:i+2, j:j+2) .^ (Q + 1))) / sum(sum(padding_img(i:i+2, j:j+2) .^ Q));
    end
end
subplot(236)
imshow(new_img(1:256, 1:256))
title('Q为正值时的逆调和均值算法')

%% 验证带阻滤波器等频域滤波器实现仅有噪声退化的图像复原
im = imread('D:\学习\数字图像处理\图片\Fig0.tif');
R_gaussian = imnoise(im, 'gaussian');
R_saltpepper = imnoise(im, 'Salt & Pepper');
R_poisson = imnoise(im, 'poisson');
% 理想带阻滤波器
[m, n] = size(R_gaussian);
F = fft2(R_gaussian);
F_shift = ifftshift(F);
d0 = 200;
h = zeros(m, n);
freq_img = zeros(m, n);
W = 80;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        if(d < d0 - W / 2)
            h = 1;
        elseif(d >= d0 - W / 2 && d <= d0 + W / 2)
            h = 0;
        else
            h = 1;
        end
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(131)
imshow(J1); title('理想带阻滤波器')

% 巴特沃斯带阻滤波器
[m, n] = size(R_gaussian);
F = fft2(R_gaussian);
F_shift = ifftshift(F);
d0 = 200;
h = zeros(m, n);
freq_img = zeros(m, n);
W = 80;
N = 6;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1./(1 + ((d.* W) ./ (d.^2 - d0.^2))^(2 * N));
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(132)
imshow(J1); title('巴特沃斯带阻滤波器')

% 高斯带阻滤波器
[m, n] = size(R_gaussian);
F = fft2(R_gaussian);
F_shift = ifftshift(F);
d0 = 200;
h = zeros(m, n);
freq_img = zeros(m, n);
W = 80;
N = 6;
for i = 1:m
    for j = 1:n
        d = sqrt((i - round(m / 2))^2 + (j - round(n / 2))^2);
        h = 1 - exp(-1/2 * ((d.^2 - d0.^2) / d .* W)^2);
        freq_img(i, j) = h * F_shift(i, j);
    end
end
J1 = ifft2(ifftshift(freq_img));
subplot(133)
imshow(J1); title('高斯带阻滤波器')


%% 由退化函数加噪声共同作用下的退化图像进行复原
% checkerboard产生测试板图像，第一个参数是每个正方形一边的像素数，第二个参数行数，第三为列数（缺省则等于行数）
%f = [1, 2, 3;4, 5, 6; 12, 13, 14];                       % 产生一个一面为8个正方形的测试板
f = im2double(R_gaussian);
PSF = fspecial('motion',7,45);                           % 运动模糊，PSF刚好为空间滤波器
gb = imfilter(f,PSF,'circular');                         % 减少边界效应
noise = imnoise(zeros(size(f)),'gaussian',0,0.001);      % 高斯噪声
g = gb + noise;                                          % 添加高斯噪声构造退化的图像模型

subplot(2,2,1);imshow(pixeldup(f,8),[ ]);title('原图像'); % 大图像运算过慢，故选用小图像来节省时间，   
subplot(2,2,2);imshow(gb);title('运动模糊图像');          % 以显示为目的，可通过像素赋值来放大图像。      
subplot(2,2,3);imshow(noise,[ ]);title('高斯噪声图像');
subplot(2,2,4);imshow(g);title('运动模糊+高斯噪声');


%f = [1, 20, 3, 40;4, 50, 6, 30;0, 20, 4, 60];                       % 产生一个一面为8个正方形的测试板
f = im2double(R_gaussian);
subplot(235);   imshow(f, []);  title('原图像');                     
PSF = fspecial('motion',3, 9);                           % 运动模糊，PSF刚好为空间滤波器
gb = imfilter(f,PSF,'circular');                         % 减少边界效应
noise = imnoise(zeros(size(f)),'gaussian',0,0.001);      % 高斯噪声
g = gb + noise;                                          % 添加高斯噪声构造退化的图像模型

fr1 = deconvwnr(g,PSF);                                  % 直接逆滤波

Sn = abs(fft(noise)).^2;                                 % 噪声功率谱
nA = sum(Sn(:))/numel(noise);                            % 平均噪声功率，prod计算数组元素的连乘积。
Sf = abs(fft2(f)).^2;                                    % 图像功率谱
fA = sum(Sf(:))/numel(f);                                % 平均图像功率
R = nA/fA;                                               % 噪信功率比
fr2 = deconvwnr(g,PSF,R);                                % 参数维纳滤波器

NCORR = fftshift(real(ifft2(Sn)));                       % 自相关函数
ICORR = fftshift(real(ifft2(Sf)));  
fr3 = deconvwnr(g,PSF,NCORR,ICORR);

subplot(2,3,1);imshow(g,[]);title('噪声图像'); 
subplot(2,3,2);imshow(fr1, []);title('逆滤波结果');          
subplot(2,3,3);imshow(fr2, []);title('使用常数比率的维纳滤波的结果');
subplot(2,3,4);imshow(fr3, []);title('使用自相关函数的维纳滤波的结果');