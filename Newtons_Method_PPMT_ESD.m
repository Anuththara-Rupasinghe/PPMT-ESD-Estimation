%% This function is used to evaluate the posterior mode and covariance of the E-step in the EM Algorithm

% Inputs:
    % spikes: the ensemble mean of spikes
    % L: the number of independent realizations of spikes
    % A: the matrix A
    % w_prev: w_k_k_1
    % Sigma_prev: P_k_k_1
    % J: the number of random variables
    % iter_Newton: the number of Newton's iterations
% Outputs:
    % w: estimate of w_k_k
    % Sigma: estimate of P_k_k
    
function [w, Sigma] = Newtons_Method_PPMT_ESD(spikes_avg, L, A, w_prev, Sigma_prev, J, iter_Newton)

%Initializing the parameters

W = size(A, 1); % Window length
w = zeros(size(w_prev)); 
lambda = 0.5*ones(W,J);
Sigma_inv = Sigma_prev \ eye(length(Sigma_prev));
V = zeros(length(w)/J, J); % Matrix form of w
gradient = zeros(length(w), 1); % Gradient vector of the objective function

for j=1:iter_Newton

    %The gradient vector
    grad_matrix = L * A' * (spikes_avg - lambda); 
    for c = 1: length(w)/J
        gradient((c-1)*J + 1 : (c)*J, 1) =  grad_matrix(c, :)';
    end       
    gradient = gradient - Sigma_inv * (w - w_prev);

    %The Hessian matrix of the objective function
    Hessian = zeros(length(w), length(w));
    for m = 1:J
        H_m = -L*A'*diag( lambda(:,m).*(1-lambda(:,m)) )*A;
        for p = 1:length(w)/J
            for q = 1:length(w)/J
                 Hessian(J*(p-1) + m, J*(q-1) + m) = H_m(p,q);
            end
        end
    end
    Hessian = Hessian - Sigma_inv;

    % Newton's update
    w = w - Hessian \ gradient;

    for p = 1: length(w)/J
    V(p, :) = (w((p-1)*J + 1 : p*J))';
    end

    x = A*V;
    lambda = 1./(1+exp(-x));
        
end

% Estimate the posterior covariance by the negative inverse of the Hessian
Sigma = -Hessian \ eye(length(w));

