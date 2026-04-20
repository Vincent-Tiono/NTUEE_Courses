clear all; clc;

%%
b = 0.09 * conv([1,-2,1], [1,2,1]);
a = conv([1,-0.6,0.25],[1,-0.2,0.02]);
figure(1);
subplot(4,2,1);
zplane(b,a);

%%
[r, p, k] = residuez(b, a);
n = 0:20;
hn = r(1)*p(1).^n + r(2)*p(2).^n + r(3)*p(3).^n + r(4)*p(4).^n;
hn(1) = hn(1)+k;
subplot(4,2,2);
stem(n, hn);
title('h[n] vs n');
xlabel('n');
ylabel('h[n]');

%%
[H,w] = freqz(b,a);

subplot(4,2,3);
plot(w,mag2db(abs(H)));
title('|H(e^j^\omega)| (dB) vs w');
xticks([0, pi/2, pi]);
xticklabels({'0', '\pi/2', '\pi'});
xlabel('w');
ylabel('|H(e^j^\omega)| (dB)');

subplot(4,2,4);
plot(w,angle(H)*360/2/pi);
title('\angleH(e^j^\omega) vs w');
xticks([0, pi/2, pi]);
xticklabels({'0', '\pi/2', '\pi'});
xlabel('w');
ylabel('\angleH(e^j^\omega');

%%
sos = zp2sos([1, 1, -1, -1], p, 0.09);
display(sos);

%%
b1 = sos(1, 1:3);
a1 = sos(1, 4:6);
[H1,w1] = freqz(b1,a1);

subplot(4,2,5);
plot(w,mag2db(abs(H1)));
title('|H_1(e^j^\omega)| (dB) vs w');
xticks([0, pi/2, pi]);
xticklabels({'0', '\pi/2', '\pi'});
xlabel('w');
ylabel('|H_1(e^j^\omega)| (dB)');

b2 = sos(2, 1:3);
a2 = sos(2, 4:6);
[H2,w2] = freqz(b2,a2);

subplot(4,2,6);
plot(w,mag2db(abs(H2)));
title('|H_2(e^j^\omega)| (dB) vs w');
xticks([0, pi/2, pi]);
xticklabels({'0', '\pi/2', '\pi'});
xlabel('w');
ylabel('|H_2(e^j^\omega)| (dB)');

subplot(4,2,7);
plot(w,mag2db(abs(H1).*abs(H2)));
title('|H_1(e^j^\omega)|\times|H_2(e^j^\omega)| (dB) vs w');
xticks([0, pi/2, pi]);
xticklabels({'0', '\pi/2', '\pi'});
xlabel('w');
ylabel('|H_1(e^j^\omega)|\times|H_2(e^j^\omega)| (dB)');

disp("Comparing (c) and (e)")
compareFunctions(mag2db(abs(H)), mag2db(abs(H1).*abs(H2)))
disp(" ")

%%
xn = zeros(1,21);
xn(1) = 1;
yn = filter(b,a,xn);
subplot(4,2,8)
stem(n, yn);
title('y[n] vs n');
xlabel('n');
ylabel('y[n]');

disp("Comparing (b) and (f)")
compareFunctions(hn, yn)


%%
function compareFunctions (x1, x2)
    len = length(x1);
    identical = true;
    for i = 1:len
        if abs(x1(i)-x2(i)) > 1e-10
            identical = false;
            break;
        end
    end

    if identical
        disp("The functions are identical.")
    else
        disp("The functions are not identical.");
    end   
end