%%
% ���ʵ�ֶ�άͼ�����ɢ����Ҷ�任������Ҷ���任
clear
clc
% ������̫�󣬶���ʽ�㲻����
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig2.tif');
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
title('��ʽ��ʵ��fft')

subplot(122)
F = fft2(A);
image(real(F))
colormap(gray)
title('matlab�Դ�fft')
% �������
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
[m, n, c] = size(im);
Gm = zeros(m);
Gn = zeros(n);

% ��ʼ��Gm
for i = 1:m
    for j = 1:m
        Gm(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / m);
    end
end
% ��ʼ��Gn
for i = 1:n
    for j = 1:n
        Gn(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / n);
    end
end
% ��ͼƬ������ͨ���ֱ�������Ҷ�任
Fouier_img_mt_1 = Gm * double(im(:, :, 1)) * Gn;
Fouier_img_mt_2 = Gm * double(im(:, :, 2)) * Gn;
Fouier_img_mt_3 = Gm * double(im(:, :, 3)) * Gn;
Fouier_img_mt = cat(3, Fouier_img_mt_1, Fouier_img_mt_2, Fouier_img_mt_3);
subplot(121)
imshow(Fouier_img_mt)
title('������ʵ��fft')

% matlab�Դ���fft2
subplot(122)
F = fft2(im);
imshow(F);
title('matlab�Դ�fft')

Real = abs(Fouier_img_mt);
Normalization_Real = (Real - min(min(Real))) ./ (max(max(Real)) - min(min(Real))) * 255;
imshow(Normalization_Real)

% iFFt
% ��������
G3 = inv(Gm);
G4 = inv(Gn);
% ��ͼƬ������ͨ���ֱ�������Ҷ���任
iFouier_img_1 = G3 * Fouier_img_mt_1 * G4 / 255;
iFouier_img_2 = G3 * Fouier_img_mt_2 * G4 / 255;
iFouier_img_3 = G3 * Fouier_img_mt_3 * G4 / 255;
iFouier_img_mt = cat(3, iFouier_img_1, iFouier_img_2, iFouier_img_3);
imshow(iFouier_img_mt)
title('����Ҷ���任');
%%
% ����fft2 ()����ʵ��һ��ͼ��ĸ���Ҷ���任����ʹ��subplot����������subimage����
% ����ʵ��ԭͼ��Ƶ��ͼ�ĶԱ���ʾ��
F = fft2(im);
subplot(221); imshow(im); title('ԭͼ��')
subplot(222); imshow(F); title('����Ҷ�任Ƶ��ͼ')

%%
% ����ifft2 ()����ʵ��Ƶ��ͼ��ĸ���Ҷ���任����ʹ��subplot����������subimage����
% ����ʵ��Ƶ��ͼ��ԭͼ��ĶԱ���ʾ��
f = ifft2(F);
k = ifft2(f) / 255;
subplot(223); imshow(im); title('����Ҷ���任')
subplot(224); imshow(k); title('����Ҷ���任Ƶ��ͼ')

%%
% �Ա��Զ���ʵ�ֵĸ���Ҷ�任������Matlab����fft2 ()��ͬһͼ��ת���Ľ���Լ�ִ��ʱ��
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
t1 = cputime;
[m, n, c] = size(im);
Gm = zeros(m);
Gn = zeros(n);

% ��ʼ��Gm
for i = 1:m
    for j = 1:m
        Gm(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / m);
    end
end
% ��ʼ��Gn
for i = 1:n
    for j = 1:n
        Gn(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / n);
    end
end
% ��ͼƬ������ͨ���ֱ�������Ҷ�任
Fouier_img_mt_1 = Gm * double(im(:, :, 1)) * Gn;
Fouier_img_mt_2 = Gm * double(im(:, :, 2)) * Gn;
Fouier_img_mt_3 = Gm * double(im(:, :, 3)) * Gn;
Fouier_img_mt = cat(3, Fouier_img_mt_1, Fouier_img_mt_2, Fouier_img_mt_3);
t2 = cputime;
t = t2 - t1;
% �Զ��帵��Ҷ�任ʱ��
t 
% �Դ�����Ҷ�任ʱ��
t1 = cputime;
F = fft2(im);
t2 = cputime;
t = t2 - t1;
t

%%
% �Ա��Զ���ʵ�ֵĸ���Ҷ���任������Matlab����ifft2 ()��ͬһͼ��ת���Ľ���Լ�ִ��ʱ�䡣
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
t1 = cputime;
[m, n, c] = size(im);
Gm = zeros(m);
Gn = zeros(n);
% ��ʼ��Gm
for i = 1:m
    for j = 1:m
        Gm(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / m);
    end
end
% ��ʼ��Gn
for i = 1:n
    for j = 1:n
        Gn(i, j) = exp(-1i * 2 * pi * (i-1) * (j-1) / n);
    end
end
% iFFt
% ��������
G3 = inv(Gm);
G4 = inv(Gn);
% ��ͼƬ������ͨ���ֱ�������Ҷ���任
iFouier_img_1 = G3 * Fouier_img_mt_1 * G4 / 255;
iFouier_img_2 = G3 * Fouier_img_mt_2 * G4 / 255;
iFouier_img_3 = G3 * Fouier_img_mt_3 * G4 / 255;
iFouier_img_mt = cat(3, iFouier_img_1, iFouier_img_2, iFouier_img_3);
t2 = cputime;
t = t2 - t1;
% �Զ��巴�任ʱ��
t

% matlab���任ʱ��
t1 = cputime;
f = ifft2(F);
t2 = cputime;
t = t2 - t1;
t

%%
% ����fftshift()��ifftshift()�Ⱥ�����ʹ�á�
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig2.tif');
subplot(221)
imshow(im);title('ԭͼ')
F = fft2(im);
subplot(222)
imshow(F); title('����Ҷ�任Ƶ��ͼ')

% fftshift
Shift_F = fftshift(F);
imshow(Shift_F)
A = abs(Shift_F);
Normalization_F = (A - min(min(A))) ./ (max(max(A)) - min(min(A))) * 255;
subplot(223)
imshow(Normalization_F)
title('ԭ�����')

% ifftshift
i_Shift_F = ifftshift(Shift_F);
AI = abs(i_Shift_F);
Normalization_iF = (AI - min(min(AI))) ./ (max(max(AI)) - min(min(AI))) * 255;
subplot(224)
imshow(Normalization_iF)
title('ԭ�����ܷ�ɢ')

%%
% ���ո���ҶƵ�׵õ������׺���λ�׵Ĵ�������
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\Fig4.bmp');
F = fft2(im);
Shift_F = fftshift(F);
sF = log(abs(Shift_F));   %��ø���Ҷ�任�ķ�����
phF = log(angle(Shift_F) * 180 / pi);   %��ø���Ҷ�任����λ��
subplot(121);
imshow(sF,[]); %��ʾͼ��ķ����ף�������[]��Ϊ�˽�sA��ֵ��������
title('����Ҷ�任������');
subplot(122);
imshow(phF,[]); %��ʾͼ����Ҷ�任����λ��
title('����Ҷ�任����λ��');

%%
% �Բ�ͬƵ�ʵ����ҹ�դ��ͼ�����Ƶ��ת�����Ա����ҹ�դ��ͼ��������׼���λ��֮������Թ�ϵ��
% ���Ȳ������ҹ�դ����
subplot(331)
I = zeros(512, 512);
for i = 1:512
    for j = 1:512
        I(i, j) = 127 + 126*cos(2 * pi * j/ 128);
    end
end
I_low = mat2gray(I);
I_low = I_low(:,:,1);
imshow(I_low); title('��Ƶͼ��')

subplot(332)
I = zeros(512, 512);
for i = 1:512
    for j = 1:512
        I(i, j) = 127 + 126*cos(2 * pi * j/ 64);
    end
end
I_middle = mat2gray(I);
I_middle = I_middle(:,:,1);
imshow(I_middle); title('��Ƶͼ��')

subplot(333)
I = zeros(512, 512);
for i = 1:512
    for j = 1:512
        I(i, j) = 127 + 126*cos(2 * pi * j/ 16);
    end
end
I_high = mat2gray(I);
I_high = I_high(:,:,1);
imshow(I_high); title('��Ƶͼ��')

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

subplot(334);imshow(s_low,[]); title('��Ƶͼ�������');
subplot(337);imshow(ph_low,[]);title('��Ƶͼ����λ��');
subplot(335);imshow(s_middle,[]);title('��Ƶͼ�������');
subplot(338);imshow(ph_middle,[]);title('��Ƶͼ����λ��');
subplot(336);imshow(s_high,[]);title('��Ƶͼ�������');
subplot(339);imshow(ph_high,[]);title('��Ƶͼ����λ��');