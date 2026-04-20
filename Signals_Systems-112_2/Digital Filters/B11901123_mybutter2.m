clear all; clc;

%%
n = 1:1000; 
Ts = 0.002; 
fs = 1/Ts; 
f1 = 100; 
M = 1000; 
xn = cos(2*pi*(n-1)*Ts) + 2*cos(2*pi*f1*(n-1)*Ts); 
 
figure(1); 
subplot(3,1,1); 
plot(n,xn); 
title('x[n] vs n'); 
xlabel('n'); 
ylabel('x[n]'); 
 
%%
[b1,a1] = butter(16,0.15); 
display(b1);
display(a1);
yn = filter(b1,a1,xn); 
subplot(3,1,2); 
plot(n,yn); 
title('y[n] vs n'); 
xlabel('n'); 
ylabel('y[n]'); 
 
%%
[b2,a2] = butter(16,[0.15,0.5],'bandpass'); 
display(b2);
display(a2);
yn = filter(b2,a2,xn); 
subplot(3,1,3); 
plot(n,yn); 
title('y[n] vs n'); 
xlabel('n'); 
ylabel('y[n]');