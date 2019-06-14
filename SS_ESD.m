%% This function generates the MAP Estimates of the latent Process, given Spiking Observations and window length(W)

function MAP_estimates = SS_ESD(Spikes, W)
%% initializing parameters 

K = size(Spikes,1); % total time duration
L = size(Spikes,2); % number of neurons
J = size(Spikes,3); % number of random processes considered
M = K / W; % total number of windows

iter_Newton = 5; % number of newton iterations per EM iteration

iter_EM = 31; % number of EM iterations

MAP_estimates = zeros(K, J);
 
%% SS-PSD MAP Estimation of the latent random variables

% Do for each random process
for signal = 1:J
    
    Spikes_avg = mean(squeeze(Spikes(:,:,signal)),2); % PSTH of the Spikes
    
    % Do for each window
    for m = 1:M
        
        % Initializing parameters and variables
        Spikes_avg_window = Spikes_avg((m-1)*W + 1 : m*W); % PSTH corresponding to the window under consideration
        x_k_K = 0*ones(W,1);     %k|K
        x_k_k = 0*ones(W,1);     %k|k
        x_k_k_1 = 0*ones(W,1);   %k|k-1         
        P_k_K = zeros(W,1);     % k|K
        P_k_k = P_k_K;          %k|k
        P_k_k_1 = P_k_K;        %k|k-1
        P_k_1_k_K = P_k_K;      %k-1,k|K
        B = zeros(W, 1);
        % Parameters
        Q = 10^(0);      % Updated in the maximization     
        alpha = 0.99;    % Fixed
        
        % EM Algorithm for the mth window of the considered signal
        for r = 1:iter_EM
             
        % % %********************************* E step ***********************************
                
            % Forward Filter
            for k = 1:W
                if k == 1
                    x_k_k_1(k) = alpha*0; 
                    P_k_k_1(k) = Q * (1 - alpha^2)^-1;
                else
                    x_k_k_1(k) = alpha * x_k_k(k-1); 
                    P_k_k_1(k) = alpha^2 * P_k_k(k-1) + Q;
                end
                [x_k_k(k), ~] = Newtons_Method_SS_ESD(Spikes_avg_window(k), L, x_k_k_1(k), P_k_k_1(k), iter_Newton);
                P_k_k(k) = 1 / ( 1/P_k_k_1(k) +  exp(x_k_k(k)));
            end
            
            % Backward Smoothing
            x_k_K = x_k_k; P_k_K = P_k_k;
            for k = W-1:-1:1
            B(k) = P_k_k(k) * (alpha) * (1 / P_k_k_1(k+1));
            x_k_K(k) = x_k_k(k) + B(k)*(x_k_K(k+1)-x_k_k_1(k+1));
            P_k_K(k) = P_k_k(k) + B(k)^2 *(P_k_K(k+1)-P_k_k_1(k+1)); 
            end
           
            %covariance smoothing
            for k = 2:W
                P_k_1_k_K(k) = P_k_K(k)*B(k-1);
            end
            
        % % %********************************* M step ***********************************    

            W_k_k1 = P_k_1_k_K + x_k_K .* [1; x_k_K(1: end -1)];
            W_k = P_k_K + x_k_K.^2;

            Q = (sum(W_k(2:end) - 2*alpha*W_k_k1(2:end) + alpha^2 * W_k(1:end-1)) + W_k(1)) / W;

        end
         
        MAP_estimates((m-1)*W + 1 : m*W, signal) = x_k_K;
         
    end
    
end

