im = imread('D:\学习\数字图像处理\图片\bear.jpg');
figure;imshow(im);
R = im(:,:,1);
G = im(:,:,2);
B = im(:,:,3);
[row,col]=size(R); %获取行数 和 列数
%%
%图像的水平错切
R_1 = zeros(row+round(b*col),col);
G_1 = zeros(row+round(b*col),col);
B_1 = zeros(row+round(b*col),col);
a=pi/6; %水平错切30度
b=tan(a);
tic
for m=1:row
    for n=1:col
        R_1(round(m+b*n),n)=R(m,n);
        G_1(round(m+b*n),n)=G(m,n);
        B_1(round(m+b*n),n)=B(m,n);
    end
end
toc
new = cat(3, R_1, G_1, B_1);
figure,imshow(uint8(new));

%%
%垂直错切30度
R_1 = zeros(col,row+round(b*col));
G_1 = zeros(col,row+round(b*col));
B_1 = zeros(col,row+round(b*col));
a=pi/6; %垂直错切30度
b=tan(a);
tic
for m=1:row
    for n=1:col
        R_1(n, round(m+b*n))=R(m,n);
        G_1(n, round(m+b*n))=G(m,n);
        B_1(n, round(m+b*n))=B(m,n);
    end
end
toc
new = cat(3, R_1, G_1, B_1);
figure,imshow(uint8(new));

%%
%图像平移变换
tic
move_rgb(im, 100, 100);
toc

