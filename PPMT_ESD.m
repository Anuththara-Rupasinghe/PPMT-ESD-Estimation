%% This function evaluates the Proposed ESD Estimate - PPMT-ESD estimate

% Inputs: 
    % Spikes:       The ensemble of Spiking Observations
    % W:            The window length
    % N:            The number of frequency bins considered in the [0, fs/2)
                    % interval
    % N_max:        The number of freqency bins estimated (smaller than N)
    % NW:           The time-half bandwidth product of multitapering
    % no_of_tapers: The number of tapers considered in multitapering
    % alpha:        Scaling of the state transition matrix
    % rho:          The scaling of the prior distribution on the parameter
                    % estimated (Q)
    % iter_EM:      The number of EM iterations performed
    % iter_Newton:  Number of Newton's iterations per EM iteration
    % fs:           The sampling frequency
    
%Output:
    % PPMT_ESD_est: The proposed ESD Estimate

function PPMT_ESD_est = PPMT_ESD(Spikes, W, N, N_max, NW, no_of_tapers, alpha, rho, iter_EM, iter_Newton, fs)
%% initializing parameters 

K = size(Spikes,1); % total time duration
L = size(Spikes,2); % number of neurons
J = size(Spikes,3); % number of random processes considered
M = K / W;          % total number of windows
[Sequences, ~] = dpss(W, NW, no_of_tapers); % Tapers used in multitapering

% calculating the PSTH and re-arranging window-wise
PSTH = mean(Spikes,2);
Spikes_avg = zeros(W, M, J); % ensemble average
for j = 1:J
    for m = 1:M
         Spikes_avg(:, m, j) = PSTH((m-1)*W + 1 : m*W, 1, j);
    end
end
x_est = log(1./(1./Spikes_avg - 1)); % Proposed Estimator for the latent variable
%% initializing the algorithm

% constructing matrix A
A=zeros(W, 2*N_max, M);
for m = 1:M
    for w=1:W
        for n=1:N_max
            A(w,2*n-1,m)=cos(((m-1)*W + w)*pi*(n-1)/N);
            A(w,2*n,m)=-sin(((m-1)*W + w)*pi*(n-1)/N);
        end
    end
end
A=2*pi*A/N;
A(:,2, :)=[];

% Initializing the variables
PPMT_ESD_est_taper = zeros(J,J, M, N_max,iter_EM, no_of_tapers);
PPMT_ESD_est_taper_dc_corrected = zeros(J,J, M+1, N_max, no_of_tapers);
PPMT_ESD_est = zeros(J,J,M+1, N_max);

% Do for each taper
for taper = 1: no_of_tapers

    % Initializing variables used in filtering and smoothing
    w_k_k = 0*ones(J*(2*N_max - 1),M);                      %k|k
    w_k_k_1 = 0*ones(J*(2*N_max - 1),M);                    %k|k-1 
    P_k_K = zeros(J*(2*N_max - 1), J*(2*N_max - 1), M);     %k|K
    for m = 1:M
        P_k_K(:,:,m) = 10^(0)*eye(J*(2*N_max - 1));
    end
    P_k_k = P_k_K;                 %k|k 
    P_k_k_1 = P_k_K;               %k|k-1
    P_k_1_k_K = P_k_K;             %k-1,k|K
    B = zeros(J*(2*N_max - 1),J*(2*N_max - 1),M);
    
    % Initializing the parameters estimated
    Q_initial_factor = 2*10^(-2); 
    iterations_for_maximization = 30;
    Q = zeros(J*(2*N_max - 1), J*(2*N_max - 1), M); % Q is estimated and updated through the EM Alogrithm
    for m = 1:M
        Q(:,:,m) = Q_initial_factor*eye(J*(2*N_max - 1));  
    end
    phi = alpha*eye(J*(2*N_max - 1));  % Note that we have fixed phi in this simulation
    
    Sequences(:, taper) = Sequences(:, taper) * 20;
    
    % Proposed estimator for the tapered ensemble mean
    Spikes_avg_tapered = Spikes_avg;    
    for j = 1:J
        for m = 1:M
            for w = 1:W    
                if Spikes_avg(w,m,j)~= 0 && Spikes_avg(w,m,j)~= 1
                        Spikes_avg_tapered(w,m,j) = 1./(1 + exp(-1* x_est(w,m,j) .* Sequences(w, taper)));
                end
            end
        end
    end
    
    %% EM Algorithm

    % rth EM iteration
    for r = 1:iter_EM

    %*********************************    E step     ***********************************
    
        fprintf('This is the EM iteration %d on taper %d.\n', r, taper);
               
        % Forward Filtering
        for m = 1:M
            if m == 1
                % one step mode prediction
                w_k_k_1(:,1) = phi*zeros(size(w_k_k_1(:,1))); 
                % one step covariance prediction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                P_k_k_1(:,:,1) =  Q(:,:,1);
            else
                % one step mode prediction
                w_k_k_1(:,m) = phi * w_k_k(:,m-1); 
                % one step covariance prediction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                P_k_k_1(:,:,m) = phi * P_k_k(:,:,m-1) * phi' + Q(:,:,m);
            end
            % Posteror mode and posterior covariance
            [w_k_k(:,m),P_k_k(:,:,m)] = Newtons_Method_PPMT_ESD(squeeze(Spikes_avg_tapered(:, m, :)), L, squeeze(A(:,:, m)), w_k_k_1(:,m), P_k_k_1(:,:,m), J, iter_Newton);
        end

        % Backward Smoothing
        w_k_K = w_k_k; P_k_K = P_k_k;
        for m = M-1:-1:1
        B(:,:,m) = P_k_k(:,:,m) * (phi') * (P_k_k_1(:,:,m+1)\eye(J*(2*N_max - 1)));
        % w(k|K)
        w_k_K(:,m) = w_k_k(:,m) + B(:,:,m)*(w_k_K(:,m+1)-w_k_k_1(:,m+1));
        % P(k|K)
        P_k_K(:,:,m) = P_k_k(:,:,m) + B(:,:,m)*(P_k_K(:,:,m+1)-P_k_k_1(:,:,m+1))*(B(:,:, m)'); 
        end

        % Covariance smoothing
        for m= 2:M
            P_k_1_k_K(:,:,m) = P_k_K(:,:,m)'*B(:,:,m-1)';
        end

    %*******************************    M step (Updating Q)    ********************************** 

        % Maximimzing the parameters of the first time window
        P_1 = P_k_K(:,:,1) + w_k_K(:,1)*w_k_K(:,1)';
        Q(:,:,1) = diag(maximization_EM_PPMT_ESD(diag(P_1), rho, iterations_for_maximization, Q_initial_factor, J, N_max));
        % Maximimzing the parameters of the following time windows
        for m = 2:M
           P_m = P_k_K(:,:,m) + w_k_K(:,m)*w_k_K(:,m)' - (P_k_1_k_K(:, :,m) + w_k_K(:, m)*w_k_K(:,m-1)')*phi' - phi * (P_k_1_k_K(:,:,m)' + w_k_K(:, m-1)*w_k_K(:, m)') + phi * (P_k_K(:,:, m-1) + w_k_K(:, m-1)*w_k_K(:, m-1)') * phi';
           Q(:,:,m) = diag(maximization_EM_PPMT_ESD(diag(P_m), rho, iterations_for_maximization, Q_initial_factor, J, N_max));
        end

        P_k_1_k_K(:,:,1) = Q(:,:,1);

        % Formulating the ESD matrix derived using the estimates of the
        % rth EM iteration
        for m = 1 :M
            R_m = (P_k_K(:,:, m) + w_k_K(:, m) * w_k_K(:, m)');
            PPMT_ESD_est_taper(:,:,m,1,r,taper) = R_m(1:J, 1:J);
            for n = 1: (N_max - 1)
                R_m_n =  R_m(J*(2*n-1) + 1: J*(2*n + 1), J*(2*n-1) + 1: J*(2*n + 1));
                PPMT_ESD_est_taper(:,:,m,n+1,r,taper) =  complex(R_m_n(1:J,1:J)+R_m_n(J+1:2*J, J+1:2*J), ( R_m_n(J+1: 2*J,1:J) - R_m_n(1:J, J+1:2*J)));
            end
        end

     end
    %% DC correction - we do a dc correction on the estimates to reduce the effect of the dominant dc component in the frequency estimates
    
    % Computation of the ESD estimates of a dc signal corresponding to the
    % considered taper
    [PSD_est_dc_taper, w_dc] = dc_correction_PPMT_ESD(N, N_max, L, J, no_of_tapers, NW, W, alpha, rho, taper);

    % Performing the dc correction on the PPMT_ESD estimate
    for m = 1:M 
        for n = 1: (N_max - 1)
            w_n = squeeze(w_k_K(J*(2*n-1) + 1: J*(2*n + 1), m)); w_dc_n = squeeze(w_dc(J*(2*n-1) + 1: J*(2*n + 1),end));
            w_j_complex = complex(w_n(1:J), w_n(J+1:2*J));  w_dc_j_complex = complex(w_dc_n(1:J), w_dc_n(J+1:2*J)); 
            PPMT_ESD_est_taper_dc_corrected(:,:,m,n+1,taper) =  PPMT_ESD_est_taper(:,:,m,n+1,end,taper) + PSD_est_dc_taper(:,:,end,n+1,end) - w_dc_j_complex * w_j_complex' - w_j_complex * w_dc_j_complex';
        end
    end
    
end

%% Proposed Final Multitaper PPMT-ESD Estimate 

for taper = 1:no_of_tapers
      PPMT_ESD_est = PPMT_ESD_est + PPMT_ESD_est_taper_dc_corrected(:,:,:,:,taper);
end

PPMT_ESD_est = (2*pi / fs)*(pi / N) * (W / N) * PPMT_ESD_est / no_of_tapers;