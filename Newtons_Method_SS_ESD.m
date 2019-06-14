%% This function is a supporting function for the SS-ESD MAP Estimate calculation. 
% This function is used to derive the forward filter in the SS-PSD estimation.

function [x, Sigma] = Newtons_Method_SS_ESD(spikes, L, x_prev, Sigma_prev, iter_Newton)

%Initializing the parameters
x = 0;
lamda = 0.5; % CIF estimate
g = L*(spikes-lamda); %gradient
Sigma_inv = 1 / Sigma_prev;
H = -L*lamda.*(1-lamda) - Sigma_inv; %Hessian

% Newtons iterations
for j=1:iter_Newton    
    x = x - g / H;
    lamda = 1/(1+exp(-x));
    g = L*(spikes - lamda) - Sigma_inv * (x - x_prev);
    H = -L* lamda*(1-lamda) - Sigma_inv;    
end

Sigma = - 1 / H ; % negative inverse of the Hessian

end