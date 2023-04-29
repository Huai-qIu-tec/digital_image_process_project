function mt = move_rgb(g,a,b)
% 图像平移函数
% g为输入RGB图像，mt为平移后的RGB图像
% a为沿水平方向的平移量
% b为沿垂直方向的平移量
% made by cgl

[m,n,l]=size(g);
p=m+abs(a);
q=n+abs(b);
jx=zeros(p,q,l);

for i=1:m
    for j=1:n
        if a>=0 && b>=0
            jx(a+i,b+j,:)=g(i,j,:);
        elseif a>=0 && b<0
            jx(a+i,j,:)=g(i,j,:);
        elseif a<0 && b>=0
            jx(i,b+j,:)=g(i,j,:);
        elseif a<0 && b<0
            jx(i,j,:)=g(i,j,:);
        end    
    end
end

mt=uint8(jx);

figure;subplot(1,2,1),imshow(g);
subplot(1,2,2),imshow(mt);