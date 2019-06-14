%% This function is used to perform a dc correction on the estimates. The goal here is to simulate spikes from a dc process,
% and calculate the PPMT Estimates of that dc process on each taper. We use those
% estimates to perform a dc correction on the original problem. Note that
% while this function is very similar to the function PPMT_ESD, this
% evaluates the spectral estimates only on a single taper, unlike PPMT_ESD.

% Inputs: the inputs are similar to the inputs to the function PPMT_ESD.m

function [ESD_est_taper_dc, w_k_K] = dc_correction_PPMT_ESD(N, N_max, L, J, no_of_tapers, NW, W, alpha, rho, taper)

iter_Newton_dc_correction = 6; % number of newton iterations per EM iteration
iter_EM_dc_correction = 16; % number of EM iterations
[Sequences, ~] = dpss(W, NW, no_of_tapers); % Tapers used in multitapering

% Use a smaller window length to speed up the algorithm
M = 3;
K = M * W;

% Simulating an esemble of Spikes from an AR process induded by a dc
% signal

x1_dc = -5.5; 
x1_true = x1_dc*ones(1,K);
sigma_n = 0.1;
x1 =  x1_true + sigma_n*randn(size(x1_true));
lamda_x1 = 1 ./ (1 + exp(-1*x1));
spikes_x1 = zeros(L,K);
for i = 1:L
spikes_x1(i,:) = rand(1,K) < lamda_x1;
end

x2_dc = x1_dc; 
x2_true = x2_dc*ones(1,K);
x2 =  x2_true + sigma_n*randn(size(x2_true));
lamda_x2 = 1 ./ (1 + exp(-1*x2));
spikes_x2 = zeros(L,K);
for i = 1:L
spikes_x2(i,:) = rand(1,K) < lamda_x2;
end

x3_dc = x1_dc;
x3_true = x3_dc*ones(1,K);
x3 =  x3_true + sigma_n*randn(size(x3_true));
lamda_x3 = 1 ./ (1 + exp(-1*x3));
spikes_x3 = zeros(L,K);
for i = 1:L
spikes_x3(i,:) = rand(1,K) < lamda_x3;
end

Spikes = zeros(K, L, J);
Spikes(:, :, 1) = spikes_x1';  
Spikes(:, :, 2) = spikes_x2'; 
Spikes(:, :, 3) = spikes_x3'; 

% calculating the PSTH and re-arranging window-wise
PSTH_temp = mean(Spikes,2);
Spikes_avg = zeros(W, M, J);
for j = 1:J
    for m = 1:M
         Spikes_avg(:, m, j) = PSTH_temp((m-1)*W + 1 : m*W, 1, j);
    end
end
x_est = log(1./(1./Spikes_avg - 1)); % Proposed estimate for the latent process

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
ESD_est_taper_dc = zeros(J,J, M, N_max,iter_EM_dc_correction, no_of_tapers);

%% Initializing the variables and parameters
 
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
Q = zeros(J*(2*N_max - 1), J*(2*N_max - 1), M);
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

for r = 1:iter_EM_dc_correction

%*********************************    E step     ***********************************

    fprintf('This is the EM iteration %d on taper %d, in dc correction.\n', r, taper);

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
        [w_k_k(:,m),P_k_k(:,:,m)] = Newtons_Method_PPMT_ESD(squeeze(Spikes_avg_tapered(:, m, :)), L, squeeze(A(:,:, m)), w_k_k_1(:,m), P_k_k_1(:,:,m), J, iter_Newton_dc_correction);
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

%*********************************     M step (Updating Q)    ***********************************    

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
    % rth EM iteration, for dc correction
    for m = 1 :M
        R_m = (P_k_K(:,:, m) + w_k_K(:, m) * w_k_K(:, m)');
        ESD_est_taper_dc(:,:,m,1,r,taper) = R_m(1:J, 1:J);
        for n = 1: (N_max - 1)
            R_m_n =  R_m(J*(2*n-1) + 1: J*(2*n + 1), J*(2*n-1) + 1: J*(2*n + 1));
            ESD_est_taper_dc(:,:,m,n+1,r,taper) =  complex(R_m_n(1:J,1:J)+R_m_n(J+1:2*J, J+1:2*J), ( R_m_n(J+1: 2*J,1:J) - R_m_n(1:J, J+1:2*J)));
        end
    end

 end