%% This script produces the results of Section V.A: Case 1: Latent trivariate process observed through spiking

close all;
clc;

%% Initializing parameters of the simulation

J = 3;          % number of different random processes (This is a trivariate case)

L = 20;         % number of realizations per CIF

% Basic signal properties
fs = 32;        % Sampling frequency
T = 2000;       % Total time duration
K = fs*T;       % Total number of samples
W = fs * 100;   % Assuming stationarity for a window of length 4 s.
M = K / W;      % The total number of windows
k = 0:K-1;

N = 800;        % Number of frequency bins considered in the frequency interval [0,fs/2). This determines the spectral resolution of the results.  

% Generate the latent Processes and the ensemble of Spiking Observations
[Signals, Spikes, True_ESD] = generate_signals(J, K, fs, W, L, N);

% Computes the Average Spiking rate. This is around 0.28 Spikes/second in
% this simulated data
Average_spiking_rate_per_neuron = zeros(L, J);
for l = 1:L
    for j = 1:J
        Average_spiking_rate_per_neuron(l, j) = length(find(squeeze(Spikes(:, l, j)) > 0)) * fs / K;
    end
end
Average_spiking_rate_per_CIF = mean(Average_spiking_rate_per_neuron, 1);

%% Plots of the latent random variables and their corresponding Spike rasters

%The three latent processes generated and their corresponding raster plots
figure(1);
subplot(6,1,1);
plot(k/fs, Signals(:,1));
title('Signal X_1');
axis([0 K/fs -15 5]);
subplot(6,1,2);
imagesc(k/fs,1:L, 1- (squeeze(Spikes(:,:,1)))');
axis([0 K/fs 0 L]);
colormap(gray);
title('Spike raster plot of signal X_1');
subplot(6,1,3);
plot(k/fs, Signals(:,2));
title('Signal X_2');
axis([0 K/fs -15 5]);
subplot(6,1,4);
imagesc(k/fs,1:L, 1- (squeeze(Spikes(:,:,2)))');
axis([0 K/fs 0 L]);
colormap(gray);
title('Spike raster plot of signal X_2');
subplot(6,1,5);
plot(k/fs, Signals(:,3));
title('Signal X_3');
axis([0 K/fs -15 5]);
subplot(6,1,6);
imagesc(k/fs,1:L, 1- (squeeze(Spikes(:,:,3)))');
axis([0 K/fs 0 L]);
colormap(gray);
title('Spike raster plot of signal X_3');

% Plotting a zoomed segment of signal x_1, and the corresponding raster
% plot
start_time = 1600; end_time = 1630;
k1 = (start_time * fs + 1) : 1 : (end_time *fs);
figure(2);
subplot(2,1,1);
plot(k1/fs, Signals(k1,1));
title('Signal X_1 from t = 1600 to t = 1630');
xlabel('Time / (s)');
grid on;
axis([start_time end_time -10 0]);
subplot(2,1,2);
imagesc(k1/fs,1:L, 1- (squeeze(Spikes(k1,:,1)))');
axis([start_time end_time 0 L]);
colormap(gray);
title('Raster plot of signal X_1 from t = 1600 to t = 1630');
xlabel('Time / (s)');

%%  Deriving the PSD estimates

NW = 2;             % time half-bandwidth product of Multitapering
no_of_tapers = 3;   % the number of tapers considered for Multitapering
N_max = N / 8;      % number of desired frequency samples which tends to be much 
                        % lower than N in neural signals because of oversampling 
overlap = 0;        % Overlapping of the sliding window has been set to zero, for ease of comparison of results

% Proposed PSD Estimate
alpha = 0.4;        % hyper-parameter: determines the scaling of the state transition matrix (state transition matrix = alpha * identity) 
rho =  2 * 10^(-1); % hyper-paramter: determines the weight of the prior on parameter Q (the estimated covariance matrix)
iter_Newton = 8;    % number of newton iterations per EM iteration
iter_EM = 16;       % number of EM iterations             
PPMT_ESD_est = PPMT_ESD(Spikes, W, N, N_max, NW, no_of_tapers, alpha, rho, iter_EM, iter_Newton, fs);

% Oracle ESD Estimate
Oracle_ESD_est = MT_ESD_matrix(Signals, W, overlap, NW, no_of_tapers, fs, N);

% SS-ESD Estimate
MAP_estimate_signals = SS_ESD(Spikes, W);
SS_ESD_est = MT_ESD_matrix(MAP_estimate_signals, W, overlap, NW, no_of_tapers, fs, N);

% PSTH-PSD Estimate
PSTH_signals = squeeze(mean(Spikes, 2));
PSTH_ESD_est = MT_ESD_matrix(PSTH_signals , W, overlap, NW, no_of_tapers, fs, N);

%% Plotting the results of spectral estimation

freq_vector = 0:0.5*fs/N:0.5*fs*(N_max-1)/N;

% Plotting the Spectrograms of the ESD estimation results in dB scale

figure(3);

% True ESD
subplot(6,5,1);
pcolor((0:M)*W/fs, freq_vector, ((squeeze(squeeze(True_ESD(1,1, :, 1:N_max))))'));
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('True ESD of X_1');
shading flat;
colormap('jet');
colorbar;
subplot(6,5,6);
pcolor((0:M)*W/fs, freq_vector, ((squeeze(squeeze(True_ESD(2,2, :, 1:N_max))))'));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('True ESD of X_2');
subplot(6,5,11);
pcolor((0:M)*W/fs, freq_vector, ((squeeze(squeeze(True_ESD(3,3, :, 1:N_max))))'));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('True ESD of X_3');
subplot(6,5,16);
pcolor((0:M)*W/fs, freq_vector, ((squeeze(squeeze(True_ESD(1,2, :, 1:N_max))))'));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross True ESD between X_1 and X_2');
subplot(6,5,21);
pcolor((0:M)*W/fs, freq_vector, ((squeeze(squeeze(True_ESD(2,3, :, 1:N_max))))'));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross True ESD between X_2 and X_3');
subplot(6,5,26);
pcolor((0:M)*W/fs, freq_vector, ((squeeze(squeeze(True_ESD(1,3, :, 1:N_max))))'));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross True ESD between X_1 and X_3');

% Oracle ESD Estimates
subplot(6,5,2);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(Oracle_ESD_est(1,1,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Oracle ESD of X_1');
subplot(6,5,7);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(Oracle_ESD_est(2,2,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Oracle ESD of X_2');
subplot(6,5,12);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(Oracle_ESD_est(3,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Oracle ESD of X_3');
subplot(6,5,17);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(Oracle_ESD_est(1,2,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross Oracle ESD between X_1 and X_2');
subplot(6,5,22);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(Oracle_ESD_est(2,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross Oracle ESD between X_2 and X_3');
subplot(6,5,27);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(Oracle_ESD_est(1,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross Oracle ESD between X_1 and X_3');

% PPMT-ESD Estimates
subplot(6,5,3);
pcolor((0:M)*W / fs,freq_vector,10*log10(abs((squeeze(squeeze(PPMT_ESD_est(1,1,:, :))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('PPMT-ESD of X_1');
subplot(6,5,8);
pcolor((0:M)*W / fs,freq_vector,10*log10(abs((squeeze(squeeze(PPMT_ESD_est(2,2,:, :))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('PPMT-ESD of X_2');
subplot(6,5,13);
pcolor((0:M)*W / fs,freq_vector,10*log10(abs((squeeze(squeeze(PPMT_ESD_est(3,3,:, :))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('PPMT-ESD of X_3');
subplot(6,5,18);
pcolor((0:M)*W / fs,freq_vector,10*log10(abs((squeeze(squeeze(PPMT_ESD_est(1,2,:, :))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross PPMT-ESD between X_1 and X_2');
subplot(6,5,23);
pcolor((0:M)*W / fs,freq_vector,10*log10(abs((squeeze(squeeze(PPMT_ESD_est(2,3,:, :))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross PPMT-ESD between X_2 and X_3');
subplot(6,5,28);
pcolor((0:M)*W / fs,freq_vector,10*log10(abs((squeeze(squeeze(PPMT_ESD_est(1,3,:, :))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross PPMT-ESD between X_1 and X_3');

% SS-ESD Estimates
subplot(6,5,4);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(SS_ESD_est(1,1,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('SS-ESD of X_1');
subplot(6,5,9);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(SS_ESD_est(2,2,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('SS-ESD of X_2');
subplot(6,5,14);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(SS_ESD_est(3,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('SS-ESD of X_3');
subplot(6,5,19);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(SS_ESD_est(1,2,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross SS-ESD between X_1 and X_2');
subplot(6,5,24);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(SS_ESD_est(2,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross SS-ESD between X_2 and X_3');
subplot(6,5,29);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(SS_ESD_est(1,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross SS-ESD between X_1 and X_3');

% PSTH-ESD Estimates
subplot(6,5,5);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(PSTH_ESD_est(1,1,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('PSTH-ESD of X_1');
subplot(6,5,10);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(PSTH_ESD_est(2,2,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('PSTH-ESD of X_2');
subplot(6,5,15);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(PSTH_ESD_est(3,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('PSTH-ESD of X_3');
subplot(6,5,20);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(PSTH_ESD_est(1,2,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross PSTH-ESD between X_1 and X_2');
subplot(6,5,25);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(PSTH_ESD_est(2,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross PSTH-ESD between X_2 and X_3');
subplot(6,5,30);
pcolor((0:M)*W*(1-overlap) / fs,freq_vector,10*log10(abs((squeeze(squeeze(PSTH_ESD_est(1,3,:, 1:N_max))))')));
shading flat;
colormap('jet');
colorbar;
xlabel('Time(s)','Interpreter','Latex');
ylabel('Frequency(Hz)','Interpreter','Latex');
title('Cross PSTH-ESD between X_1 and X_3');


% Plotting the snapshot of the spectrogram at the 8th time window (t = 700
% s to t = 800 s).
selected_window = 8;

figure(4);
subplot(3,1,1)
plot(freq_vector, squeeze(squeeze(squeeze(True_ESD(1,1, selected_window, 1:N_max)))),freq_vector, 10*log10(abs(squeeze(squeeze(Oracle_ESD_est(1,1, selected_window, 1:N_max))))), freq_vector, 10*log10(abs(squeeze(squeeze(PPMT_ESD_est(1,1, selected_window, :))))), freq_vector, 10*log10(abs(squeeze(squeeze(SS_ESD_est(1,1, selected_window, 1:N_max))))),freq_vector, 10*log10(abs(squeeze(squeeze(PSTH_ESD_est(1,1, selected_window, 1:N_max))))));
grid on;
title('ESD of X_1');
xlabel('frequency / Hz','Interpreter','Latex');
ylabel('Amplitude / dB','Interpreter','Latex');
legend('True ESD', 'Oracle ESD', 'PPMT-ESD','SS-ESD','PSTH-ESD');
subplot(3,1,2)
plot(freq_vector, squeeze(squeeze(squeeze(True_ESD(2,2, selected_window, 1:N_max)))),freq_vector, 10*log10(abs(squeeze(squeeze(Oracle_ESD_est(2,2, selected_window, 1:N_max))))), freq_vector, 10*log10(abs(squeeze(squeeze(PPMT_ESD_est(2,2, selected_window, :))))), freq_vector, 10*log10(abs(squeeze(squeeze(SS_ESD_est(2,2, selected_window, 1:N_max))))),freq_vector, 10*log10(abs(squeeze(squeeze(PSTH_ESD_est(2,2, selected_window, 1:N_max))))));
grid on;
title('ESD of X_2');
xlabel('frequency / Hz','Interpreter','Latex');
ylabel('Amplitude / dB','Interpreter','Latex');
% legend('True ESD', 'Oracle ESD', 'PPMT-ESD','SS-ESD','PSTH-ESD');
subplot(3,1,3)
plot(freq_vector, squeeze(squeeze(squeeze(True_ESD(1,2, selected_window, 1:N_max)))),freq_vector, 10*log10(abs(squeeze(squeeze(Oracle_ESD_est(1,2, selected_window, 1:N_max))))), freq_vector, 10*log10(abs(squeeze(squeeze(PPMT_ESD_est(1,2, selected_window, :))))), freq_vector, 10*log10(abs(squeeze(squeeze(SS_ESD_est(1,2, selected_window, 1:N_max))))), freq_vector, 10*log10(abs(squeeze(squeeze(PSTH_ESD_est(1,2, selected_window, 1:N_max))))));
grid on;
title('Cross ESD between X_1 and X_2');
xlabel('frequency / Hz','Interpreter','Latex');
ylabel('Amplitude / dB','Interpreter','Latex');
% legend('True ESD', 'Oracle ESD', 'PPMT-ESD','SS-ESD','PSTH-ESD');

%% Computing normalized Mean Squared Error (MSE) of the estimates in dB scale 

% We remove the dc component in error calculations due to the dominant
% error induced by it in all ESD estimates
start = 2;

mse_Oracle_ESD = sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)- 10*log10(abs(Oracle_ESD_est(:,:, 1:M, start:N_max)))).^2)))) / sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)).^2))));
mse_SS_ESD = sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)- 10*log10(abs(SS_ESD_est(:,:, 1:M, start:N_max)))).^2)))) / sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)).^2))));
mse_PSTH_ESD = sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)- 10*log10(abs(PSTH_ESD_est(:,:, 1:M, start:N_max)))).^2)))) / sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)).^2))));
mse_PPMT_ESD = sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)- 10*log10(abs(PPMT_ESD_est(:,:, 1:M, start:end)))).^2)))) / sum(sum(sum(sum((True_ESD(:,:,1:M,start:N_max)).^2))));
