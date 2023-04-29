%% �����˲� --> Ƶ���˲�
% ƽ���˲���
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
subplot(131)
imshow(f);        title('ԭͼ')
h = fspecial('average', [5 5]);
AvgBlur = imfilter(f, h, 'replicate');
subplot(132)
imshow(AvgBlur);    title('ƽ���˲���');
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
imshow(AvgFreqzBlur);    title('Ƶ��ƽ���˲�');
figure;
subplot(121)
imshow(abs(H), []);     title('Ƶ�������');
subplot(122)
imshow(abs(H1),[]);     title('Ƶ�����ɢ');

% ��
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
subplot(131)
imshow(f);        title('ԭͼ')
h = fspecial('sobel');
AvgBlur = imfilter(f, h, 'replicate');
subplot(132)
imshow(f - AvgBlur);    title('���˲���');
PQ = paddedsize(size(f));
H = freqz2(h, PQ(1), PQ(2));
H1 = ifftshift(H);
AvgFreqzBlur = dftfilt(f, H);
subplot(133)
imshow(f - AvgFreqzBlur);    title('Ƶ�����˲�');

%%
% ��ͨ�˲���
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
subplot(221)
imshow(f);  title('ԭͼ��');
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
%ʱ��ͼ��ת��ΪƵ��
K1=J1.*H1;
L1=ifft2(ifftshift(K1));           
%Ƶ��ͼ��ת��ΪʱƵ
L1=L1(1:size(I,1), 1:size(I, 2));
subplot(222);imshow(L1);
title('�����ͨ�˲���');
n=6;
H2=1./(1+(D./D0).^(2*n));
J2=fftshift(fft2(I, size(H2, 1), size(H2, 2)));  %ʱ��ͼ��ת��ΪƵ��
K2=J2.*H2;
L2=ifft2(ifftshift(K2));
%Ƶ��ͼ��ת��ΪʱƵ
L2=L2(1:size(I,1), 1:size(I, 2));
subplot(223);imshow(L2);
title('������˹��ͨ�˲���');
H3=exp(-(D.^2)./(2*(D0.^2)));
J3=fftshift(fft2(I, size(H3, 1), size(H3, 2)));  %ʱ��ͼ��ת��ΪƵ��
K3=J3.*H3;
L3=ifft2(ifftshift(K3));
%Ƶ��ͼ��ת��ΪʱƵ
L3=L3(1:size(I,1), 1:size(I, 2));
subplot(224);imshow(L3);
title('��˹��ͨ�˲���');

%% ��ͬ����Ƶ���µ��˲���
% d0 = 10ʱ��ʵ�õ�ͨ�˲���
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 10��ʵ�õ�ͨ�˲���')
% d0 = 20ʱ��ʵ�õ�ͨ�˲���
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
imshow(J1); title('d0 = 20��ʵ�õ�ͨ�˲���')
% d0 = 50ʱ��ʵ�õ�ͨ�˲���
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
imshow(J1); title('d0 = 50��ʵ�õ�ͨ�˲���')

% d0 = 100ʱ��ʵ�õ�ͨ�˲���
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
imshow(J1); title('d0 = 100��ʵ�õ�ͨ�˲���')

% d0 = 10������˹��ͨ�˲�
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 10�İ�����˹��ͨ�˲���')

% d0 = 20������˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 20�İ�����˹��ͨ�˲���')

% d0 = 50������˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 50�İ�����˹��ͨ�˲���')

% d0 = 100������˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 100�İ�����˹��ͨ�˲���')

% d0 = 10��˹��ͨ�˲�
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 10�ĸ�˹��ͨ�˲���')

% d0 = 20��˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 20�ĸ�˹��ͨ�˲���')

% d0 = 50��˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 50�ĸ�˹��ͨ�˲���')

% d0 = 100��˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(J1); title('d0 = 100�ĸ�˹��ͨ�˲���')

%% ��ͨ�˲���
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
subplot(221)
imshow(f);  title('ԭͼ��');
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
%ʱ��ͼ��ת��ΪƵ��
K1 = J1.*H1;
L1 = ifft2(ifftshift(K1));           
%Ƶ��ͼ��ת��ΪʱƵ
L1 = L1(1:size(I,1), 1:size(I, 2));
subplot(222);imshow(im2double(f) - L1);
title('�����ͨ�˲���');
n=6;
H2 = 1./(1+(D0./D).^(2*n));
J2 = fftshift(fft2(I, size(H2, 1), size(H2, 2)));  %ʱ��ͼ��ת��ΪƵ��
K2 = J2.*H2;
L2 = ifft2(ifftshift(K2));
%Ƶ��ͼ��ת��ΪʱƵ
L2 = L2(1:size(I,1), 1:size(I, 2));
subplot(223);imshow(im2double(f) - L2);
title('������˹��ͨ�˲���');
H3 = 1 - exp(-(D.^2)./(2*(D0.^2)));
J3 = fftshift(fft2(I, size(H3, 1), size(H3, 2)));  %ʱ��ͼ��ת��ΪƵ��
K3 = J3.*H3;
L3 = ifft2(ifftshift(K3));
%Ƶ��ͼ��ת��ΪʱƵ
L3 = L3(1:size(I,1), 1:size(I, 2));
subplot(224);imshow(im2double(f) - L3);
title('��˹��ͨ�˲���');
%% �����ͨ�˲���
% d0 = 10ʱ��ʵ�ø�ͨ�˲���
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 10��ʵ�ø�ͨ�˲���')
% d0 = 20ʱ��ʵ�õ�ͨ�˲���
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
imshow(im2double(f) - J1); title('d0 = 20��ʵ�ø�ͨ�˲���')
% d0 = 50ʱ��ʵ�õ�ͨ�˲���
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
imshow(im2double(f) - J1); title('d0 = 50��ʵ�ø�ͨ�˲���')

% d0 = 100ʱ��ʵ�õ�ͨ�˲���
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
imshow(im2double(f) - J1); title('d0 = 100��ʵ�ø�ͨ�˲���')
%% ������˹��ͨ�˲���
% d0 = 10������˹��ͨ�˲�
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 10�İ�����˹��ͨ�˲���')

% d0 = 20������˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 20�İ�����˹��ͨ�˲���')

% d0 = 50������˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 50�İ�����˹��ͨ�˲���')

% d0 = 100������˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 100�İ�����˹��ͨ�˲���')
%% ��˹��ͨ�˲���
% d0 = 10��˹��ͨ�˲�
f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 10�ĸ�˹��ͨ�˲���')

% d0 = 20��˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 20�ĸ�˹��ͨ�˲���')

% d0 = 50��˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 50�ĸ�˹��ͨ�˲���')

% d0 = 100��˹��ͨ�˲�

f = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig0.tif');
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
imshow(im2double(f) - J1); title('d0 = 100�ĸ�˹��ͨ�˲���')