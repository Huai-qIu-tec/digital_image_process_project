%% 1.imratio的使用
im = imread('D:\学习\数字图像处理\图片\Fig2.tif');
im1 = imread('D:\学习\数字图像处理\图片\Fig1.tif');
subplot(121);    imshow(im)     
subplot(122);    imshow(im1)
r = imratio(im, im1);
disp(r)


%% 2.compare的使用
f=imread('rice.png');
[X,map]=gray2ind(f,16);
Gray_image=ind2gray(X,map);
subplot(1,2,1),imshow(f);
subplot(1,2,2),imshow(Gray_image);
compare(f,Gray_image)

%% 信源熵
im = imread('D:\学习\数字图像处理\图片\Fig2.tif');
tic
p = hist(double(im), 256);
p = p / sum(p);
H = -sum(p(p ~= 0) .* log2(p(p ~= 0)))
toc
%% 霍夫曼编码和解码
im = imread('D:\学习\数字图像处理\图片\Fig2.tif');
f=uint8([2 3 4 2; 3 2 4 4; 2 2 1 2; 1 1 2 2]);
c = huffman(hist(double(f(:)), 4))
h1f2=c(f(:))'
h2f2=char(h1f2)'

f = imread('rice.png');
c = mat2huff(f);
cr1 = imratio(f,c);

g=huff2mat(c);
imshow(uint8(g))
rmse = compare(f,g);