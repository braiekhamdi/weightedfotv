function [Dx, Dy] = applyD(u, a, alpha, R)
% Apply discrete Caputo gradient: backward quadrature on second differences.
% u: Nx x Ny
[Nx, Ny] = size(u);
h = 1; % assume unit spacing; the factor h^-alpha can be absorbed in a
scale = 1 / gamma_func(2-alpha);  % 1/Gamma(2-alpha) factor
scale = scale * (1); % h^{-alpha} with h=1
% Preallocate
Dx = zeros(Nx, Ny);
Dy = zeros(Nx, Ny);

% For Dx: for each pixel (i,j) compute sum_{m=0}^{i-1} a_m * Delta_x^2 u_{i-m,j}
% We'll compute truncated sums; handle boundaries via mirror padding
u_pad = padarray(u,[1 1],'symmetric'); % simple pad to safely index centered differences

% Precompute second-differences along x and y for full domain (interior)
% For centered second diff we need neighbors; handle via symmetric extension
% We'll compute Delta_x2 array aligned with original indices
Delta_x2 = zeros(Nx, Ny);
for i = 1:Nx
    ip = min(i+1, Nx);
    im = max(i-1, 1);
    Delta_x2(i,:) = u(ip,:) - 2*u(i,:) + u(im,:);
end
Delta_y2 = zeros(Nx, Ny);
for j = 1:Ny
    jp = min(j+1, Ny);
    jm = max(j-1, 1);
    Delta_y2(:,j) = u(:,jp) - 2*u(:,j) + u(:,jm);
end

% apply convolution-like sums (backward)
for i = 1:Nx
    % m from 0 to i-1 (capable of truncation)
    maxm = min(R-1, i-1);
    % vector of coefficients and indices
    if maxm >= 0
        idxs = i - (0:maxm);
        coeffs = a(1:(maxm+1)); % a_m maps to MATLAB index m+1
        % sum Delta_x2(idxs,:)*coeffs
        % perform weighted sum along rows
        tmp = zeros(1, Ny);
        for t = 1:length(idxs)
            tmp = tmp + coeffs(t) * Delta_x2(idxs(t), :);
        end
        Dx(i,:) = scale * tmp;
    end
end

for j = 1:Ny
    maxn = min(R-1, j-1);
    if maxn >= 0
        idxs = j - (0:maxn);
        coeffs = a(1:(maxn+1));
        tmp = zeros(Nx,1);
        for t=1:length(idxs)
            tmp = tmp + coeffs(t) * Delta_y2(:, idxs(t));
        end
        Dy(:,j) = scale * tmp;
    end
end

% multiply by h^{-alpha} (here h=1 so omitted)
end
