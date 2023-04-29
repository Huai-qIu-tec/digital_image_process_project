function y = jihe(matrix)
[m, n] = size(matrix);
y = zeros(1, n);
for i = 1:n
    y(i) = prod(prod(matrix(:, i)))^(1 / (m * n));
end
end