im = imread('D:\ѧϰ\����ͼ����\ͼƬ\bear.jpg');
figure;imshow(im);
R = im(:,:,1);
G = im(:,:,2);
B = im(:,:,3);
[row,col]=size(R); %��ȡ���� �� ����
%%
%ͼ���ˮƽ����
R_1 = zeros(row+round(b*col),col);
G_1 = zeros(row+round(b*col),col);
B_1 = zeros(row+round(b*col),col);
a=pi/6; %ˮƽ����30��
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
%��ֱ����30��
R_1 = zeros(col,row+round(b*col));
G_1 = zeros(col,row+round(b*col));
B_1 = zeros(col,row+round(b*col));
a=pi/6; %��ֱ����30��
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
%ͼ��ƽ�Ʊ任
tic
move_rgb(im, 100, 100);
toc

