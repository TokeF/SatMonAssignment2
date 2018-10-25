function T = calcT(M)
svv = M.sVV; shh = M.sHH; shv = M.sHV;
wSize = 5;

t11 = abs(shh + svv).^2;
h = fspecial('average', wSize);
t11 = imfilter(t11,h);

t12 = abs( (shh + svv) .* conj((shh-svv)) ).^2;
h = fspecial('average', wSize);
t12 = imfilter(t12,h);

t13 = abs( (shh + svv) .* conj(shv) ).^2;
h = fspecial('average', wSize);
t13 = 2*imfilter(t13,h);

t21 = (shh-svv) .* conj((shh + svv)) ;
h = fspecial('average', wSize);
t21 = imfilter(t21,h);

t22 = abs(shh - svv).^2;
h = fspecial('average', wSize);
t22 = imfilter(t22,h);

t23 = (shh - svv) .* conj(shv);
h = fspecial('average', wSize);
t23 = 2*imfilter(t23,h);

t31 = shv .* conj(shh + svv);
h = fspecial('average', wSize);
t31 = 2*imfilter(t31,h);

t32 = shv .* conj(shh - svv);
h = fspecial('average', wSize);
t32 = 2*imfilter(t32,h);

t33 = abs(shv).^2;
h = fspecial('average', wSize);
t33 = 4*imfilter(t33,h);

T = cell(2800,2800);
for i = 1: 2800
    for j = 1 : 2800
        tij = [t11 t12 t13;
            t21 t22 t23;
            t31 t32 t33];
        T{i,j} = tij;
    end
end
end