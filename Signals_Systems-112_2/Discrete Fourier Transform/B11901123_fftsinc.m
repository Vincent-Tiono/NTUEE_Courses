clear all; clc;

%%
N = 1001; 
T = 100; 
N1 = (N-1)/2; 
Ts = T/N1; 
n = -N1:N1; 
t = -N1*Ts: Ts: N1*Ts; 
xn = sin(2*pi*t) ./ (2*pi*t); 
xn(N1+1) = 1; 
 
figure(1); 
plot(n,xn); 
title('x[n] vs n'); 
xlabel('n'); 
ylabel('x[n]'); 
 
%%
X = fftshift(fft(xn)); 
f = -pi:2*pi/(N-1):pi; 
 
figure(2); 
plot(f, abs(X)); 
xticks([-pi, -pi/2, 0, pi/2, pi]); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
title('|X(e^j^\omega)| vs \omega (fft)'); 
xlabel('\omega'); 
ylabel('|X(e^j^\omega)|'); 
 
%%
Xk = zeros(1,N); 
for k = -N1:N1; 
    Xk(k+N1+1) = sum(xn .* exp(-1i*2*pi*n*k/N)); 
end 
 
figure(3); 
plot(f, abs(Xk)); 
xticks([-pi, -pi/2, 0, pi/2, pi]); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
title('|X_k(e^j^\omega)| vs \omega (formula)'); 
xlabel('\omega'); 
ylabel('|X_k(e^j^\omega)|'); 

%%
Tw = T/2; 
wn = zeros(1,N); 
for i = 1:N; 
    if(n(i)*Ts>=-Tw/2 && n(i)*Ts<=Tw/2) 
        wn(i) = 0.5*(1+cos(2*pi*abs(n(i)*Ts)/Tw)); 
    end 
end 
 
figure(4); 
plot(n,wn); 
title('w[n]'); 
xlabel('n'); 
ylabel('w[n]'); 
 
%%
yn = xn .* wn; 
 
figure(5); 
plot(n,yn); 
title('y[n]'); 
xlabel('n'); 
ylabel('y[n]'); 
 
%%
Y = fftshift(fft(yn)); 
 
figure(6); 
plot(f, abs(Y)); 
xticks([-pi, -pi/2, 0, pi/2, pi]); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
title('|Y(e^j^\omega)| vs \omega (fft)'); 
xlabel('\omega'); 
ylabel('|Y(e^j^\omega)|');


% Verification
figure(7)
plot(f, abs(X)-abs(Xk))
xticks([-pi, -pi/2, 0, pi/2, pi]); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
title('|X(e^j^\omega)| - |X_k(e^j^\omega)| vs \omega '); 
xlabel('\omega'); 
ylabel('|X(e^j^\omega)| - |X_k(e^j^\omega)|'); 

W = fftshift(fft(wn)); 
 
figure(8); 
plot(f, abs(W)); 
xticks([-pi, -pi/2, 0, pi/2, pi]); 
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'}); 
title('|W(e^j^\omega)| vs \omega (fft)'); 
xlabel('\omega'); 
ylabel('|W(e^j^\omega)|');
