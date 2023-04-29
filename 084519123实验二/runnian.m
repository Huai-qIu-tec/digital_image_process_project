function flag = runnian(year)
flag = 0;
if(mod(year, 400) == 0)
    flag = 1;
elseif(mod(year, 4) == 0 && mod(year, 100) ~= 0)
    flag = 1;
else
    flag = 0;
end
end