%% This function produces the classical sliding window multitaper estimate of the Evolutionary Spectral Density Matrix of a given multivariate random process

% Inputs:
    %Signals: `     The multivariate process
    %W:             Window length of stationarity
    %Overlap:       Percentage overlapping between sliding windows
    %NW:            Time half bandwidth of multitapering
    %no_of_tapers:  Number of tapers considered for multitapering
    %fs:            Sampling Frequency
    %N:             Total number of frequency bins
    
%Output:
    %ESD_est:       Multitaper Evolutionary Spectral Density Matrix
                    %Estimate
    
function ESD_est = MT_ESD_matrix(Signals, W, overlap, NW, no_of_tapers, fs, N)

% Initializing variables
J = size(Signals, 2); % This is a J-variate random process
W_new = W*(1-overlap); % Window length with overlap        
M = floor(length(Signals)/W_new - W/W_new); % Number of windows

Fourier_est = zeros(2*N, M+1, no_of_tapers, J); % Eigen Spectra
ESD_est = zeros(J, J, M+2, 2*N); % Multitaper Spectral Density Matrix
[dpss_seq,~] = dpss(W, NW, no_of_tapers); 

% Computing the mutivariate Eigen Spectra for each taper
for sig = 1:J
for k = 0:1:M
    d = squeeze(Signals(1+k*W_new:k*W_new+W,sig)); 
    for taper = 1:no_of_tapers
        Fourier_est(:, k+1, taper, sig) = fft((d - mean(d)).*dpss_seq(:,taper), 2*N); % Eigen spectra of the tapered process
    end
end
end

% Computing the Multitaper Spectral Density Matrix Estimate
for i = 1:J
    for j = 1:J
        for k = 0:1:M
            temp = zeros(size(Fourier_est,1),1);
            for taper = 1:no_of_tapers
               temp  = temp + squeeze(squeeze(squeeze(Fourier_est(:, k+1, taper, i)))) .* conj(squeeze(squeeze(squeeze(Fourier_est(:, k+1, taper, j)))));
            end
            ESD_est(i, j, k+1, :) = temp;
        end
    end
end

ESD_est = ESD_est /(fs * no_of_tapers); % Final Multitaper ESD Estimate

end