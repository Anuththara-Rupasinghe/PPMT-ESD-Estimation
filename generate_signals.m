%% This function generates the Signals and Spikes used in the simulation in Section V.A

%Inputs: 
    % J: The number of random processes (This is a J-variate process)
    % K: Total sample duration of the process
    % fs: Sampling frequency
    % W: Window length
    % L: Number of realization per CIF
    % N: Total number of frequency bins
%Outputs:
    % Signals: The latent random process
    % Spikes: The ensemble of Spiking observations
    % True_ESD: The theoretical ESD derived using the closed form
                % expressions for the PSD of an AR Model
                
function [Signals, Spikes, True_PSD] = generate_signals(J, K, fs, W, L, N)

T = K / fs; % Total duration in time
M = K / W; % Total number of windows
t = 0:(T/(fs*T-1)):T;

% Set of frequencies the AR components are tuned around
f0 = 0.0008;  % in Hz
f1 = 1.15; %in Hz
f2 = 1.3; % in Hz
f3 = 0.95;  % in Hz
f4 = 1.5;  % in Hz
f5 = 0.65;  % in Hz
f6 = 1.85;  % in Hz

%% Generation of the latent random processes

% Input signal for the AR filters
sigma_input = 1.0 * 10^-6; 
input = randn(size( 0:((T+100)/(fs*(T+100)-1)):T+100));
input = input - mean(input);

% First AR Component - tuned around f1 (Note that this is also amplitude modulated around
% frequeny f_0)
a1 = .064;
b1 = conv(conv(conv([1 -0.99*exp(1j*2*(pi)*f1/fs)],[1 -0.99*exp(-1j*2*(pi)*f1/fs)]),conv([1 -0.99*exp(1j*2*(pi)*f1/fs)],[1 -0.99*exp(-1j*2*(pi)*f1/fs)])),conv([1 -0.99*exp(1j*2*(pi)*f1/fs)],[1 -0.99*exp(-1j*2*(pi)*f1/fs)]));
y1 = filter(a1,b1, sigma_input*input);
y1 = y1(end-fs*T+1:end).*((1.5*cos(2*pi*f0*t)).^8 + .17);

% Second AR Component - tuned around f2
a2 = 1.4;
b2 = conv(conv(conv([1 -0.99*exp(1j*2*(pi)*f2/fs)],[1 -0.99*exp(-1j*2*(pi)*f2/fs)]),conv([1 -0.99*exp(1j*2*(pi)*f2/fs)],[1 -0.99*exp(-1j*2*(pi)*f2/fs)])),conv([1 -0.99*exp(1j*2*(pi)*f2/fs)],[1 -0.99*exp(-1j*2*(pi)*f2/fs)]));
y2 = filter(a2,b2,sigma_input*input);
y2 = y2(end-fs*T+1:end);

% Third AR Component - tuned around f3
a3 = 0.65;
b3 = conv(conv(conv([1 -0.99*exp(1j*2*(pi)*f3/fs)],[1 -0.99*exp(-1j*2*(pi)*f3/fs)]),conv([1 -0.99*exp(1j*2*(pi)*f3/fs)],[1 -0.99*exp(-1j*2*(pi)*f3/fs)])),conv([1 -0.99*exp(1j*2*(pi)*f3/fs)],[1 -0.99*exp(-1j*2*(pi)*f3/fs)]));
y3 = filter(a3,b3, sigma_input*input);
y3 = y3(end-fs*T+1:end);

% Fourth AR Component - tuned around f4
a4 = 2.4;
b4 = conv(conv(conv([1 -0.99*exp(1j*2*(pi)*f4/fs)],[1 -0.99*exp(-1j*2*(pi)*f4/fs)]),conv([1 -0.99*exp(1j*2*(pi)*f4/fs)],[1 -0.99*exp(-1j*2*(pi)*f4/fs)])),conv([1 -0.99*exp(1j*2*(pi)*f4/fs)],[1 -0.99*exp(-1j*2*(pi)*f4/fs)]));
y4 = filter(a4,b4,sigma_input*input);
y4_1 = y4(end-fs*T+1:end);
y4_2 = y4(end-fs*T+1 - 6:end - 6);

% Fifth AR Component - tuned around f5
a5 =  0.2;
b5 = conv(conv(conv([1 -0.99*exp(1j*2*(pi)*f5/fs)],[1 -0.99*exp(-1j*2*(pi)*f5/fs)]),conv([1 -0.99*exp(1j*2*(pi)*f5/fs)],[1 -0.99*exp(-1j*2*(pi)*f5/fs)])),conv([1 -0.99*exp(1j*2*(pi)*f5/fs)],[1 -0.99*exp(-1j*2*(pi)*f5/fs)]));
y5 = filter(a5, b5, sigma_input*input);
y5 = y5(end-fs*T+1:end);
y5_mod = [zeros(1, 0.4*M*W), y5(0.4*M*W + 1: K)];

% Sixth AR Component - tuned around f6
a6 = 8.0;
b6 = conv(conv(conv([1 -0.98*exp(1j*2*(pi)*f6/fs)],[1 -0.98*exp(-1j*2*(pi)*f6/fs)]),conv([1 -0.99*exp(1j*2*(pi)*f6/fs)],[1 -0.99*exp(-1j*2*(pi)*f6/fs)])),conv([1 -0.99*exp(1j*2*(pi)*f6/fs)],[1 -0.99*exp(-1j*2*(pi)*f6/fs)]));
y6 = filter(a6, b6, sigma_input*input);
y6_1 = y6(end-fs*T+1 - 10: end - 10);
y6_2 = y6(end-fs*T+1:end);
y6_3 = [y6_1(1:K/2), y6_2(K/2 + 1: K)];

SNR = 20; % Signal to noise ratio of the generated processes

% Forming the Process x_1 by linear combinations of the above AR processes
x_1_dc = -5.5; 
factor_x_1 = 1.2;
x_1_true =  y1 + factor_x_1*y4_1 + factor_x_1*y5_mod + x_1_dc;
sigma_n_x_1 = sqrt(var(x_1_true)/10^(SNR/10));
x_1 =  x_1_true + sigma_n_x_1*randn(size(x_1_true));

% Forming the Process x_2 by linear combinations of the above AR processes
x_2_dc = -5.5; 
factor_x_2 = 0.83;
x_2_true =  factor_x_2* y3 + factor_x_2*y4_2 + factor_x_2*y5 + factor_x_2*y6_2 + x_2_dc;
sigma_n_x_2 = sqrt(var(x_2_true)/10^(SNR/10));
x_2 =  x_2_true + sigma_n_x_2*randn(size(x_2_true));

% Forming the Process x_3 by linear combinations of the above AR processes
x_3_dc = -5.5;
factor_x_3 = 1.0; 
x_3_true =  y2 + factor_x_3*y5 + factor_x_3*y6_3 + x_3_dc;
sigma_n_x_3 = sqrt(var(x_3_true)/10^(SNR/10));
x_3 =  x_3_true + sigma_n_x_3*randn(size(x_3_true));

Signals = [x_1', x_2', x_3'];
%% Generating the Spikes

% Spikes of Process X_1
lamda_x_1 = 1 ./ (1 + exp(-1*x_1));
spikes_x_1 = zeros(L,K);
avg_spiking_rate_x_1 = zeros(L,1);
for i = 1:L
spikes_x_1(i,:) = rand(1,K) < lamda_x_1;
avg_spiking_rate_x_1(i) =  length(find(spikes_x_1(i,:) > 0)) / T;
end

% Spikes of Process X_2
lamda_x_2 = 1 ./ (1 + exp(-1*x_2));
spikes_x_2 = zeros(L,K);
avg_spiking_rate_x_2 = zeros(L,1);
for i = 1:L
spikes_x_2(i,:) = rand(1,K) < lamda_x_2;
avg_spiking_rate_x_2(i) =  length(find(spikes_x_2(i,:) > 0)) / T;
end

% Spikes of Process X_3
lamda_x_3 = 1 ./ (1 + exp(-1*x_3));
spikes_x_3 = zeros(L,K);
avg_spiking_rate_x_3 = zeros(L,1);
for i = 1:L
spikes_x_3(i,:) = rand(1,K) < lamda_x_3;
avg_spiking_rate_x_3(i) =  length(find(spikes_x_3(i,:) > 0)) / T;
end

Spikes = zeros(K, L, J);
Spikes(:, :, 1) = spikes_x_1';  
Spikes(:, :, 2) = spikes_x_2'; 
Spikes(:, :, 3) = spikes_x_3'; 
%% Evaluation of the Theoretical PSD

% Evaluating the Eigen Spectra of each AR component
[Hb1,~] = freqz(a1,b1,N,fs);
[Hb2,~] = freqz(a2,b2,N,fs);
[Hb3,~] = freqz(a3,b3,N,fs);
[Hb4,~] = freqz(a4,b4,N,fs);
[Hb5,~] = freqz(a5,b5,N,fs);
[Hb6,~] = freqz(a6,b6,N,fs);
Hb1_scaled = zeros(N, M);

% Computing the ESD of each process 
Px1 = zeros(N, M+1);    % ESD of X_1
Px2 = zeros(N, M+1);    % ESD of X_2
Px3 = zeros(N, M+1);    % ESD of X_3
Px1_x2 = zeros(N, M+1); % Cross ESD between X_1 and X_2
Px1_x3 = zeros(N, M+1); % Cross ESD between X_1 and X_3
Px2_x3 = zeros(N, M+1); % Cross ESD between X_2 and X_3
for m = 1:M
    Hb1_scaled(:, m) = mean(((1.5*cos(2*pi*f0*t((m-1)*W +1 : m*W))).^8 + .17))*Hb1;
    if m > .4*M
        Px1(:, m) = 10*log10(((abs(Hb1_scaled(:,m) + factor_x_1*Hb4 + factor_x_1*Hb5)).^2 * sigma_input^2  * var(input) + sigma_n_x_1^2 )/fs);
        Px1_x3(:, m) = 10*log10(abs((((Hb1_scaled(:,m) + factor_x_1*Hb4 + factor_x_1*Hb5).*conj(Hb2 + factor_x_3*Hb5 + factor_x_3*Hb6))* sigma_input^2 * var(input) )/fs));
        Px1_x2(:, m) = 10*log10(abs((((Hb1_scaled(:,m) + factor_x_1*Hb4 + factor_x_1*Hb5).*conj(factor_x_2*Hb3 + factor_x_2*Hb4 + factor_x_2*Hb5 + factor_x_2*Hb6))* sigma_input^2 * var(input) )/fs));
    else
        Px1(:, m) = 10*log10(((abs(Hb1_scaled(:,m) + factor_x_1*Hb4)).^2 * sigma_input^2 * var(input) + sigma_n_x_1^2 )/fs);
        Px1_x3(:, m) = 10*log10(abs((((Hb1_scaled(:,m) + factor_x_1*Hb4).*conj(Hb2 + factor_x_3*Hb5 + factor_x_3*Hb6))* sigma_input^2 * var(input) )/fs));
        Px1_x2(:, m) = 10*log10(abs((((Hb1_scaled(:,m) + factor_x_1*Hb4).*conj(factor_x_2*Hb3 + factor_x_2*Hb4 + factor_x_2*Hb5 + factor_x_2*Hb6))* sigma_input^2 * var(input) )/fs));
    end
    Px3(:, m) = 10*log10(((abs(Hb2 + factor_x_3*Hb5 + factor_x_3*Hb6)).^2 * sigma_input^2 * var(input)  + sigma_n_x_2^2 )/fs);
    Px2(:, m) = 10*log10(((abs(factor_x_2*Hb3 + factor_x_2*Hb4 + factor_x_2*Hb5 + factor_x_2*Hb6)).^2 * sigma_input^2 * var(input)  + sigma_n_x_3^2 )/fs);
    Px2_x3(:, m) = 10*log10(abs((((Hb2 + factor_x_3*Hb5 + factor_x_3*Hb6).*conj(factor_x_2*Hb3 + factor_x_2*Hb4 + factor_x_2*Hb5 + factor_x_2*Hb6))* sigma_input^2 * var(input) )/fs));
end

%Forming the Evolutionary Spectral Density Matrix by combining the ESDs
True_PSD = zeros(J,J,M+1,N);
True_PSD(1,1,:,:) = Px1';
True_PSD(2,2,:,:) = Px2';
True_PSD(3,3,:,:) = Px3';
True_PSD(1,2,:,:) = Px1_x2';
True_PSD(2,1,:,:) = conj(True_PSD(1,2,:,:));
True_PSD(2,3,:,:) = Px2_x3';
True_PSD(3,2,:,:) = conj(True_PSD(2,3,:,:));
True_PSD(1,3,:,:) = Px1_x3';
True_PSD(3,1,:,:) = conj(True_PSD(1,3,:,:));
