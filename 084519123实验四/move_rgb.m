function mt = move_rgb(g,a,b)
% ͼ��ƽ�ƺ���
% gΪ����RGBͼ��mtΪƽ�ƺ��RGBͼ��
% aΪ��ˮƽ�����ƽ����
% bΪ�ش�ֱ�����ƽ����
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