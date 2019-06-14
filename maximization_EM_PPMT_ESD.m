%% This function is an implementation of Multivariate Newton Raphson for the maximization of the diagonal elements of the covariance matrix Q_m

% Inputs: 
    % p:    diagonal vector of matrix P_m
    % rho:  the hyper parameter of the prior distribution
    % iterations: the number of Newton's updates
    % factor: initialization of the vector q
    % J: the number of variables
    % N: the number of frequency bins considered
    
% Outputs:
    % q: the updates for the diagonal entries of Q_m
    
function q = maximization_EM_PPMT_ESD(p, rho, iterations, factor, J, N)

q = factor * ones(length(p), 1);
Hessian = zeros(length(p), length(p));

for r = 1 : iterations
  
    % Gradient of the objective function
    gradient = 0.5 * (1 - p ./ q);
    gradient(J+1 : 2*J*(N - 2) + J) = gradient(J+1 : 2*J*(N - 2) + J) + 2 * rho * log (q(J+1 : 2*J*(N - 2) + J) ./ q(J+1 + 2*J : 2*J*(N - 2) + J + 2*J));
    gradient(3*J + 1: 2*J*(N-1) + J) = gradient(3*J + 1: 2*J*(N-1) + J) + 2 * rho * log(q(3*J + 1: 2*J*(N-1) + J) ./ q(J + 1: 2*J*(N-2) + J));

    % Hessian of the objective function
    for n = 1 : length(p)
        Hessian(n,n) = p(n) / (2 * (q(n))^2);
        if (n >= J+1) && (2*J*(N-2) + J >= n)
            Hessian(n,n) =  Hessian(n,n) + 2 * rho / q(n);
            Hessian(n, n + 2*J) = -2 * rho / q(n + 2*J);
        end
        if (n >= 3*J + 1) && (2*J*(N-1) + J >= n)
            Hessian(n,n) = Hessian(n,n) + 2 * rho / q(n);
            Hessian(n, n - 2*J) =  -2 * rho / q(n - 2*J);
        end
    end

    % Newton's update
    q = q - Hessian \ gradient;
  
end