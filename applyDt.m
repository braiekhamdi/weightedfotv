function Dt_v = applyDt(vx, vy, a, alpha, R)
% Apply adjoint D^T to the stacked field (vx,vy); returns Nx*Ny vector (as matrix)
[Nx, Ny] = size(vx);
% Initialize
res = zeros(Nx, Ny);
scale = 1 / gamma_func(2-alpha); % 1/Gamma(2-alpha) factor

% Adjoint operation roughly: accumulate contributions from forward convolution
% For Dx adjoint: for each i, contribute to u_k via reversed coefficients and second-diff adjoint
% We implement by accumulating into res via reversed loops

% For x-direction: contributions from vx(i,:)
for i = 1:Nx
    maxm = min(R-1, i-1);
    if maxm >= 0
        coeffs = a(1:(maxm+1)); % a_0...a_maxm
        for t = 1:(maxm+1)
            idx = i - (t-1); % position of Delta_x2 used
            % Delta_x2 at idx multiplies coeffs(t); we need to add contribution to neighboring u entries
            % Delta_x2 at idx affects u_{idx-1}, u_{idx}, u_{idx+1} with weights [1, -2, 1]
            % So contribution to u_{idx-1} += coeff * vx(i,:) * 1
            %                    u_{idx}   += coeff * vx(i,:) * (-2)
            %                    u_{idx+1} += coeff * vx(i,:) * 1
            c = coeffs(t);
            if idx-1 >= 1
                res(idx-1, :) = res(idx-1, :) + c * vx(i, :);
            end
            res(idx, :) = res(idx, :) - 2 * c * vx(i, :);
            if idx+1 <= Nx
                res(idx+1, :) = res(idx+1, :) + c * vx(i, :);
            end
        end
    end
end

% For y-direction: similar accumulation
for j = 1:Ny
    maxn = min(R-1, j-1);
    if maxn >= 0
        coeffs = a(1:(maxn+1));
        for t = 1:(maxn+1)
            idx = j - (t-1);
            c = coeffs(t);
            if idx-1 >= 1
                res(:, idx-1) = res(:, idx-1) + c * vy(:, j);
            end
            res(:, idx) = res(:, idx) - 2 * c * vy(:, j);
            if idx+1 <= Ny
                res(:, idx+1) = res(:, idx+1) + c * vy(:, j);
            end
        end
    end
end

Dt_v = scale * res;
end

