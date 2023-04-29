function result = while_M(x)
if(x > 2)
    result = 2*exp(x);
elseif(x > -2 && x < 2)
    result = x^2 + 2;
elseif(x < -2)
    result = -x^2 - 2;
else
    fprintf('x不再范围内\n');
end
end

x = 0:pi/50:2*pi;
sin_y = sin(x);
cos_y = cos(x);
[hAx,hLine1,hLine2] = plotyy(x,sin_y,x,cos_y);
hLine1.LineStyle = '--';
hLine1.LineWidth = 2;
hLine2.LineStyle = '-';
hLine2.LineWidth = 2;
legend('sin(x)', 'cos(x)');
xlabel('range(x)');
title('Sin(x)与Cos(x)在2pi的取值');

x = -50:1:50;
y = -50:1:50;
[xx,yy] = meshgrid(x,y);
z = xx.^2 + yy.^2;
surf(z)
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Z = X^2 + y^2');
