%%
% ʵ�ֲ�ɫͼ���ƽ������
% �����ֶ�ʵ��padding
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig0.tif');
im = imnoise(im, 'salt & pepper',0.02);
global size_value;
size_value=inputdlg('������ģ��','ģ��',[1 40]);
size_value=str2double(char(size_value));
fp = padarray(im, [floor((size_value-1)/2) floor((size_value-1)/2)], 0, 'both');
h = fspecial('average',[size_value size_value]);
filterImg = imfilter(fp, h);
figure;
subplot(121);
imshow(im);title('����֮���ԭͼ');
subplot(122);
imshow(filterImg);title('ƽ��֮��');

% ���ú����Դ�padding���ظ����
figure;
HSI = rgb2hsi(im);
H = HSI(:,:,1);
S = HSI(:,:,2);
I = HSI(:,:,3);
subplot(131)
imshow(im);title('RGBԭͼ');
subplot(132)
imshow(HSI);title('HSIģʽ�µ�ԭͼ');
subplot(133)
im_filter_I = imfilter(I, h, 'replicate');
filterImg = hsi2rgb(cat(3, H, S, im_filter_I));
imshow(filterImg);title('��Iͨ��ƽ��֮��')

% ���ú����Դ�padding���������
figure;
im_filter = imfilter(im, h, 'symmetric');
subplot(121);
imshow(im);title('ԭͼ');
subplot(122);
imshow(im_filter);title('�������');


% ������ƽ�� -- ��ֵ�˲�
[~, ~, c] = size(im);
if c == 3
    I1 = medfilt2(im(:, :, 1), [size_value size_value]);
    I2 = medfilt2(im(:, :, 2), [size_value size_value]);
    I3 = medfilt2(im(:, :, 3), [size_value size_value]);
    I = cat(3, I1, I2, I3);
    imshow(I);title('��ֵ�˲�');
else
    I = medfilt2(im, [size_value size_value]);
    imshow(I);
end
%%
% ʵ�ֲ�ɫͼ����񻯴���
im = imread('D:\ѧϰ\����ͼ����\ͼƬ\CFig0.tif');

% Laplace����
[h, w, c] = size(im);
% �ȶ�ͼƬ����padding���ٸ���Laplace��ʽ����matlab����������������ÿ����������
img_copy = im2double(im);
img_temp = padarray(img_copy, [1 1], 'replicate');  % padding�ظ����
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
imshow(im);title('ԭͼ');
subplot(1,2,2)
imshow(laplace_img);
title('3 �� 3��Laplace����');

% Sobel����
figure;
subplot(221);
imshow(im);title('ԭͼ');
pattern_X = [-1 -2 -1;0 0 0 ;1 2 1];    % ˮƽ�����sobel����
pattern_Y = pattern_X';                 % ��ֱ�����sobel����
im_before = im2double(im);
gradX = im_before - imfilter(im_before, pattern_X, 'replicate');
subplot(222);
imshow(gradX);title('ˮƽ����Sobel��');
gradY = im_before - imfilter(im_before, pattern_Y, 'replicate');
subplot(223);
imshow(gradY);title('��ֱ����Sobel��');
grad = sqrt(gradX.^2 + gradY.^2);
subplot(224);
imshow(grad);title('Sobel��');

sobel = fspecial('sobel');
im_sobel = im_before - imfilter(im_before, sobel);
imshow(im_sobel)

% Prewitt����
figure;
subplot(221);
imshow(im);title('ԭͼ');
pattern_X = [-1 -1 -1;0 0 0 ;1 1 1];    % ˮƽ�����prewitt����
pattern_Y = pattern_X';                 % ��ֱ�����prewitt����
im_before = im2double(im);
gradX = im_before - imfilter(im_before, pattern_X, 'replicate');
subplot(222);
imshow(gradX);title('ˮƽ����Prewitt��');
gradY = im_before - imfilter(im_before, pattern_Y, 'replicate');
subplot(223);
imshow(gradY);title('��ֱ����Prewitt��');
grad = sqrt(gradX.^2 + gradY.^2);
subplot(224);
imshow(grad);title('Prewitt��');
figure;
prewitt = fspecial('prewitt');
im_prewitt = im_before - imfilter(im_before, prewitt);
imshow(im_prewitt);

% roberts����
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
imshow(im);title('ԭͼ');
subplot(122)
imshow(im_Robert);title('Roberts��ͼ');

%%
% ʵ�ֲ�ɫͼ��ı�Ե��⼰�ָ��
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
% ѧϰice������ʹ�ã�ͨ���Ķ�ice�����Ľű����������������ߵ��ں�����
z = interp1q([0 255]', [0 255]', [0:255]');

ice('image', im)
