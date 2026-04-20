clear all; clc;

%%
n = 1:100; 
fs =20; 
Ts = 1/fs; 
xn = cos(2*pi*(n-1)*Ts); 
 
figure(1); 
plot(n,xn); 
title('x[n] vs n'); 
xlabel('n'); 
ylabel('x[n]'); 
 
%%
[b1,a1] = butter(3,0.05); 
[H1,w1] = freqz(b1,a1,1000); 
yn = filter(b1,a1,xn); 
display(b1);
display(a1);
 
figure(2); 
subplot(3,3,1); 
plot(w1,mag2db(abs(H1))); 
title('|H_b(e^j^\omega)| (dB) vs w'); 
xticks([0, pi/2, pi]); 
xticklabels({'0', '\pi/2', '\pi'}); 
xlabel('w'); 
ylabel('|H_b(e^j^\omega)| (dB)'); 
 
subplot(3,3,2); 
plot(w1,angle(H1)*360/2/pi); 
title('\angleH_b(e^j^\omega) vs w'); 
xticks([0, pi/2, pi]); 
xticklabels({'0', '\pi/2', '\pi'}); 
xlabel('w'); 
ylabel('\angleH_b(e^j^\omega'); 
 
subplot(3,3,3); 
plot(n,yn); 
title('y_b[n] vs n'); 
xlabel('n'); 
ylabel('y_b[n]'); 
 
%%
[b2,a2] = butter(7,0.05); 
[H2,w2] = freqz(b2,a2,1000); 
yn = filter(b2,a2,xn); 
display(b2);
display(a2);

subplot(3,3,4); 
plot(w2,mag2db(abs(H2))); 
title('|H_c(e^j^\omega)| (dB) vs w'); 
xticks([0, pi/2, pi]); 
xticklabels({'0', '\pi/2', '\pi'}); 
xlabel('w'); 
ylabel('|H_c(e^j^\omega)| (dB)'); 
 
subplot(3,3,5); 
plot(w2,angle(H2)*360/2/pi); 
title('\angleH_c(e^j^\omega) vs w'); 
xticks([0, pi/2, pi]); 
xticklabels({'0', '\pi/2', '\pi'}); 
xlabel('w'); 
ylabel('\angleH_c(e^j^\omega'); 
 
subplot(3,3,6); 
plot(n,yn); 
title('y_c[n] vs n'); 
xlabel('n'); 
ylabel('y_c[n]'); 
 
%%
[b3,a3] = butter(3,0.5); 
[H3,w3] = freqz(b3,a3,1000); 
yn = filter(b3,a3,xn); 
display(b3);
display(a3);

subplot(3,3,7); 
plot(w3,mag2db(abs(H3))); 
title('|H_d(e^j^\omega)| (dB) vs w'); 
xticks([0, pi/2, pi]); 
xticklabels({'0', '\pi/2', '\pi'}); 
xlabel('w'); 
ylabel('|H_d(e^j^\omega)| (dB)'); 
 
subplot(3,3,8); 
plot(w3,angle(H3)*360/2/pi); 
title('\angleH_d(e^j^\omega) vs w'); 
xticks([0, pi/2, pi]); 
xticklabels({'0', '\pi/2', '\pi'}); 
xlabel('w'); 
ylabel('\angleH_d(e^j^\omega'); 
 
subplot(3,3,9); 
plot(n,yn); 
title('y_d[n] vs n'); 
xlabel('n'); 
ylabel('y_d[n]');