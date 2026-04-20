clear all; clc

n = -10:50;
m = -5:10;

% piecewise x1
a1 = zeros(size(n));
for i = 1:length(n)
    if n(i)<=20 && n(i)>=1
        a1(i)=n(i);
    elseif n(i)>=21 && n(i)<=39
        a1(i)=40-n(i);
    else
        a1(i)=0;
    end
end

% piecewise x2
b1 = zeros(size(n));
for i = 1:length(n)
    if n(i)>=1 && n(i)<=10
        b1(i) = 1;
    else
        b1(i) = 0;
    end
end

% exponential x1
a2 = zeros(size(m));
for i = 1:length(m)
    if m(i)>=1 && m(i)<=3
        a2(i) = 3.^m(i);
    else
        a2(i) = 0;
    end
end

% exponential x2
b2 = zeros(size(m));
for i = 1:length(m)
    if m(i)>=1 && m(i)<=5
        b2(i) = 2.^m(i);
    else
        b2(i) = 0;
    end
end

A1 = zeros(2*length(a1)-1,length(a1));
for i = 1:length(a1)
    for j = 1:length(a1)
        A1(i+j-1,j) = a1(i);
    end
end

A2 = zeros(2*length(a2)-1,length(a2));
for i = 1:length(a2)
    for j = 1:length(a2)
        A2(i+j-1,j) = a2(i);
    end
end

B1 = b1';
B2 = b2';

tiledlayout(4,2)

%%
a1x = nexttile;
stem(a1x,-10:50,a1)
xlabel('-10 ≤ n ≤ 50') 
ylabel('x_1[n]') 
title(a1x,'x1[n] vs n')

%%
b1x = nexttile;
stem(b1x,-10:50,b1)
xlabel('-10 ≤ n ≤ 50') 
ylabel('x_2[n]') 
title(b1x,'x2[n] vs n')

%%
convab1x = nexttile;
stem(convab1x,-20:100,conv(a1,b1))
xlabel('-20 ≤ n ≤ 100') 
ylabel('y[n]') 
title(convab1x,'y[n] vs n')

%%
convAB1x = nexttile;
stem(convAB1x,-20:100,(A1*B1)')
xlabel('-20 ≤ n ≤ 100') 
ylabel('y[n]') 
title(convAB1x,'y[n] vs n (computed by matrix product)')

%%
a2x = nexttile;
stem(a2x,-5:10,a2)
xlabel('-5 ≤ n ≤ 10') 
ylabel('x_1[n]') 
title(a2x,'Problem(d) --- x1[n] vs n')

%%
b2x = nexttile;
stem(b2x,-5:10,b2)
xlabel('-5 ≤ n ≤ 10') 
ylabel('x_2[n]')
title(b2x,'Problem(d) --- x2[n] vs n')

%%
convab2x = nexttile;
stem(convab2x,-10:20,conv(a2,b2))
xlabel('-10 ≤ n ≤ 20') 
ylabel('y[n]')
title(convab2x,'Problem(d) --- y[n] vs n')

%%
convAB2x = nexttile;
stem(convAB2x,-10:20,(A2*B2)')
xlabel('-10 ≤ n ≤ 20') 
ylabel('y[n]')
title(convAB2x,'Problem(d) --- y[n] vs n  (computed by matrix)')

%%
disp("Comparing (b) and (c)")
compareFunctions(conv(a1,b1), A1*B1)

disp("Comparing (d)-3 and (d)-4")
compareFunctions(conv(a2,b2), A2*B2)

function compareFunctions (x1, x2)
    len = length(x1);
    identical = true;
    for i = 1:len
        if ne(x1(i), x2(i))
            identical = false;
            break;
        end
    end

    if identical
        disp("The functions are identical.")
    else
        disp("The functions are not indentical.");
    end   
end