%% 1.ģ���������
% ��˹����
Gaussian_noise = imnoise2('gaussian', 10000, 1, 0.1);
subplot(241)
histogram(Gaussian_noise, 100);
title('��˹����');

% ��������
Uniform_noise = imnoise2('uniform', 10000, 1, 0.1);
subplot(242)
histogram(Uniform_noise, 100);
title('��������');

% ������̬����
Lognormal_noise = imnoise2('lognormal', 10000, 1, 1, 0.25);
subplot(243)
histogram(Lognormal_noise, 100);
title('������̬����');

% ��������
Rayleigh_noise = imnoise2('rayleigh', 10000, 1, 0.1);
subplot(244)
histogram(Rayleigh_noise, 100);
title('��������');

% ָ������
exp_noise = imnoise2('exponential', 10000, 1, 1);
subplot(245)
histogram(exp_noise, 100);
title('ָ������');

% ����������
erlang_noise = imnoise2('erlang', 10000, 1, 2, 5);
subplot(246)
histogram(erlang_noise, 100);
title('����������');

% ��������
SaltPepper_noise = imnoise2('salt & pepper', 10000, 1, 0.05, 0.05);
subplot(247)
histogram(SaltPepper_noise, 100);
title('��������');

% ��������
C = [0 64;0 128;32 32;64 0;128 0;-32 32];
[r, R, S] = imnoise3(512,512,C);
subplot(2,3,1),imshow(S,[]);title('Ƶ��') %��ʾƵ��
subplot(2,3,4),imshow(r,[]);title('��������') %��ʾ�ռ���������ģʽ
C = [0 32;0 64;16 16;32 0;64 0;-16 16]; %�ı���λ�ã��۲�Ƶ�׺Ϳռ���������ģʽ�仯 [r,R,S]=imnoise3(512,512,C);
subplot(2,3,2),imshow(S,[]);title('Ƶ��')
subplot(2,3,5),imshow(r,[]);title('��������')
C = [6 32;-2 2]; %�ı���λ�ã��۲�ռ���������ģʽ�仯
[r, R, S] = imnoise3(512,512,C);
subplot(2,3,3),imshow(r,[]);title('�ı���λ����������')
A = [1 5]; %ʹ�÷�Ĭ���������A,�۲�ռ���������ģʽ�仯
[r, R, S] = imnoise3(512,512,C,A);
subplot(2,3,6),imshow(r,[]);title('��Ĭ�����������������')
%% 2.���������ģ��������
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
[m, n, c] = size(im);
% ���ý���������ȾͼƬ
im_saltpepper = im;
R = imnoise2('salt & pepper', m, n, 0, 0.1); % ������Ⱦͼ��
index_salt = find(R == 1);
im_saltpepper(index_salt) = 255;

R = imnoise2('salt & pepper', m, n, 0.1, 0); % ������Ⱦͼ��
index_pepper = find(R == 0);
im_saltpepper(index_pepper) = 0;
subplot(231)
imshow(im_saltpepper);   title('��������ͼ');
h = fspecial('average', [3 3]);
im_saltpepper = imfilter(im_saltpepper, h, 'replicate'); 
subplot(234)
imshow(im_saltpepper);   title('��������ƽ��ͼ')

% ��Ӹ�˹����
R_gaussian = imnoise2('gaussian', m, n, 0, 0.1);
im_gaussian = im2double(im) + R_gaussian;
subplot(232)
imshow(im_gaussian);    title('��˹����ͼ')
h = fspecial('average', [3 3]);
im_gaussian = imfilter(im_gaussian, h, 'replicate'); 
subplot(235)
imshow(im_gaussian);    title('��˹����ƽ��ͼ')

% ��Ӿ�������
R_uniform = imnoise2('uniform', m, n, -0.2, 0.2);
im_uniform = im2double(im) + R_uniform;
subplot(233)
imshow(im_uniform);     title('��������ͼ')
h = fspecial('average', [3 3]);
im_uniform = imfilter(im_uniform, h, 'replicate'); 
subplot(236)
imshow(im_uniform);    title('��������ƽ��ͼ')

%% ��֤������ֵ�����ξ�ֵ�����;�ֵ������;�ֵ�ȿռ����˲���
% ͼƬ����
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
R_gaussian = imnoise(im, 'gaussian');
R_saltpepper = imnoise(im, 'Salt & Pepper');
R_poisson = imnoise(im, 'poisson');

% ������ֵ
subplot(231);   imshow(R_gaussian);     title('��˹����');
h = fspecial('average', [3 3]);
mean_filter = imfilter(R_gaussian, h, 'replicate');
subplot(234);   imshow(mean_filter);     title('��˹������ֵƽ��ͼ');

subplot(232);   imshow(R_saltpepper);     title('��˹����');
mean_filter = imfilter(R_gaussian, h, 'replicate');
subplot(235);   imshow(mean_filter);     title('����������ֵƽ��ͼ');

subplot(233);   imshow(R_poisson);     title('��˹����');
mean_filter = imfilter(R_gaussian, h, 'replicate');
subplot(236);   imshow(mean_filter);     title('����������ֵƽ��ͼ');

% ���ξ�ֵ
% ��˹����

subplot(231);    imshow(R_gaussian);     title('��˹����ͼ');
padding_img = padarray(R_gaussian, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = prod(prod(padding_img(i:i+2, j:j+2)))^(1/9);
    end
end
subplot(234);   imshow(new_img(1:256, 1:256));  title('��˹���ξ�ֵ�˲�');

% ��������

subplot(232);    imshow(R_poisson);     title('��˹����ͼ');
padding_img = padarray(R_poisson, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = prod(prod(padding_img(i:i+2, j:j+2)))^(1/9);
    end
end
subplot(235);   imshow(new_img(1:256, 1:256));  title('���ɼ��ξ�ֵ�˲�');

% ��������
subplot(233);    imshow(R_saltpepper);     title('��˹����ͼ');
padding_img = padarray(R_saltpepper, [3 3], 'replicate');
[M, N] = size(padding_img);
padding_img = im2double(padding_img);
new_img = zeros(M, N);
for i = 1:M-2
    for j = 1:N-2
        new_img(i, j) = prod(prod(padding_img(i:i+2, j:j+2)))^(1/9);
    end
end
subplot(236);   imshow(new_img(1:256, 1:256));  title('���μ��ξ�ֵ�˲�');

% ���;�ֵ ���ڡ��Ρ�����Ч���ã����������ڡ�����������
% ��˹����
subplot(231);   imshow(R_gaussian);  title('��˹����');
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
title('����ƽ����˹����');

% ��������
subplot(232);   imshow(R_saltpepper);  title('��������');
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
title('����ƽ����������');

% ��������
subplot(233);   imshow(R_poisson);  title('��������');
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
title('����ƽ����������');

% ����;�ֵ�˲��� 
Q = -1;
R_saltpepper1 = R_saltpepper;
subplot(231);   imshow(R_saltpepper1);  title('��������');
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
title('QΪ��ֵʱ������;�ֵ�㷨')

Q = 1;
R_saltpepper2 = R_saltpepper;
subplot(232);   imshow(R_saltpepper2);  title('��������');
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
title('QΪ��ֵʱ������;�ֵ�㷨')

Q = 0;
R_saltpepper3 = R_saltpepper;
subplot(233);   imshow(R_saltpepper3);  title('��������');
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
title('QΪ��ֵʱ������;�ֵ�㷨')

%% ��֤�����˲�����Ƶ���˲���ʵ�ֽ��������˻���ͼ��ԭ
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
R_gaussian = imnoise(im, 'gaussian');
R_saltpepper = imnoise(im, 'Salt & Pepper');
R_poisson = imnoise(im, 'poisson');
% ��������˲���
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
imshow(J1); title('��������˲���')

% ������˹�����˲���
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
imshow(J1); title('������˹�����˲���')

% ��˹�����˲���
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
imshow(J1); title('��˹�����˲���')


%% ���˻�������������ͬ�����µ��˻�ͼ����и�ԭ
% checkerboard�������԰�ͼ�񣬵�һ��������ÿ��������һ�ߵ����������ڶ�����������������Ϊ������ȱʡ�����������
%f = [1, 2, 3;4, 5, 6; 12, 13, 14];                       % ����һ��һ��Ϊ8�������εĲ��԰�
f = im2double(R_gaussian);
PSF = fspecial('motion',7,45);                           % �˶�ģ����PSF�պ�Ϊ�ռ��˲���
gb = imfilter(f,PSF,'circular');                         % ���ٱ߽�ЧӦ
noise = imnoise(zeros(size(f)),'gaussian',0,0.001);      % ��˹����
g = gb + noise;                                          % ��Ӹ�˹���������˻���ͼ��ģ��

subplot(2,2,1);imshow(pixeldup(f,8),[ ]);title('ԭͼ��'); % ��ͼ�������������ѡ��Сͼ������ʡʱ�䣬   
subplot(2,2,2);imshow(gb);title('�˶�ģ��ͼ��');          % ����ʾΪĿ�ģ���ͨ�����ظ�ֵ���Ŵ�ͼ��      
subplot(2,2,3);imshow(noise,[ ]);title('��˹����ͼ��');
subplot(2,2,4);imshow(g);title('�˶�ģ��+��˹����');


%f = [1, 20, 3, 40;4, 50, 6, 30;0, 20, 4, 60];                       % ����һ��һ��Ϊ8�������εĲ��԰�
f = im2double(R_gaussian);
subplot(235);   imshow(f, []);  title('ԭͼ��');                     
PSF = fspecial('motion',3, 9);                           % �˶�ģ����PSF�պ�Ϊ�ռ��˲���
gb = imfilter(f,PSF,'circular');                         % ���ٱ߽�ЧӦ
noise = imnoise(zeros(size(f)),'gaussian',0,0.001);      % ��˹����
g = gb + noise;                                          % ��Ӹ�˹���������˻���ͼ��ģ��

fr1 = deconvwnr(g,PSF);                                  % ֱ�����˲�

Sn = abs(fft(noise)).^2;                                 % ����������
nA = sum(Sn(:))/numel(noise);                            % ƽ���������ʣ�prod��������Ԫ�ص����˻���
Sf = abs(fft2(f)).^2;                                    % ͼ������
fA = sum(Sf(:))/numel(f);                                % ƽ��ͼ����
R = nA/fA;                                               % ���Ź��ʱ�
fr2 = deconvwnr(g,PSF,R);                                % ����ά���˲���

NCORR = fftshift(real(ifft2(Sn)));                       % ����غ���
ICORR = fftshift(real(ifft2(Sf)));  
fr3 = deconvwnr(g,PSF,NCORR,ICORR);

subplot(2,3,1);imshow(g,[]);title('����ͼ��'); 
subplot(2,3,2);imshow(fr1, []);title('���˲����');          
subplot(2,3,3);imshow(fr2, []);title('ʹ�ó������ʵ�ά���˲��Ľ��');
subplot(2,3,4);imshow(fr3, []);title('ʹ������غ�����ά���˲��Ľ��');