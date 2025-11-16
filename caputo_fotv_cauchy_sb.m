% caputo_fotv_cauchy_sb.m
% Split-Bregman solver for Caputo FOTV + Cauchy fidelity
%
% Usage:
%   u = caputo_fotv_cauchy_sb(f, opts)
%
% Inputs:
%   f    - noisy image (double, grayscale)
%   opts - struct with options (see default values below)
%
% Outputs:
%   u    - restored image
%
% NOTE:
%  - Implements Caputo discrete derivative for 1<alpha<2 using backward quadrature
%    with second-differences and weights a_m^(alpha) = (m+1)^(1-alpha) - m^(1-alpha).
%  - IRLS inner loop linear solve uses PCG with matrix-free matvec via applyD/applyDt.
%  - d-update: isotropic shrinkage with weight map w.
%
% Author: Hamdi Braiek (hamdi.houicht@gmail.com)
% Date: 2025-10-21
%-------------------------------------------------------------------------------

function u = caputo_fotv_cauchy_sb(f, opts)

if nargin<2, opts = struct(); end

% -------------------- default parameters --------------------
[Nx, Ny] = size(f);
default.alpha     = 1.5;      % fractional order in (1,2)
default.lambda    = 0.1;      % regularization weight
default.beta      = 5;        % SB penalty
default.gamma     = 10.0;     % Cauchy scale parameter (depends on noise)
default.maxOuter  = 100;      % outer SB iterations
default.maxInner  = 1;        % IRLS inner iterations per SB outer
default.pcgTol    = 1e-4;     % PCG tolerance
default.pcgMaxIt  = 200;      % PCG max iters
default.epsOmega  = 1e-3;    % eps for omega
default.omegaClip = [1e-3, 1e3];
default.R         = max(Nx,Ny); % truncation radius for Caputo convolution (set to full)
default.w         = ones(Nx,Ny); % weight map (spatial adaptivity) - default flat
default.mu        = 0;        % tiny stabilizer (optional)
default.tolOuter  = 1e-4;
default.verbose   = true;
% merge defaults
fn = fieldnames(default);
for k=1:numel(fn)
    if ~isfield(opts, fn{k}), opts.(fn{k}) = default.(fn{k}); end
end

alpha = opts.alpha;
lambda = opts.lambda;
beta = opts.beta;
gamma = opts.gamma;
maxOuter = opts.maxOuter;
maxInner = opts.maxInner;
pcgTol = opts.pcgTol;
pcgMaxIt = opts.pcgMaxIt;
epsOmega = opts.epsOmega;
omegaClip = opts.omegaClip;
R = opts.R;
wmap = opts.w;
mu = opts.mu;
tolOuter = opts.tolOuter;
verbose = opts.verbose;

u = f;
u(isnan(u) | isinf(u)) = 0;

% ensure double
f = double(f);
f = f - min(f(:));
f = f / max(f(:));
u = f;                         % initialization
[Nx, Ny] = size(f);



% precompute Caputo weights a_m^(alpha) for m=0..R-1
m = (0:(R-1))';
a = ( (m+1).^(1-alpha) - m.^(1-alpha) );    % a_m^(alpha)
gamma_const = gamma;

% Precompute second difference operator convenience: handled inside applyD

% initialize d and b
[Dx_u, Dy_u] = applyD(u, a, alpha, R);
d_x = Dx_u; d_y = Dy_u;
b_x = zeros(size(d_x)); b_y = zeros(size(d_y));

% precompute indices for PCG
nPixels = Nx*Ny;

% outer loop
prev_u = u;
for k = 1:maxOuter
    % IRLS inner loop(s)
    for inner = 1:maxInner
        % compute residuals and IRLS weights
        r = f - u;
        %omega = 2 ./ (gamma_const^2 + r.^2 + epsOmega);
        %omega = min(max(omega, omegaClip(1)), omegaClip(2)); % clip
        omega = 2 ./ (opts.gamma^2 + r.^2 + epsOmega);  % avoid division by zero
        omega = min(omega, 2/opts.gamma^2);         % bound upper range
        omega = max(omega, 1e-3);                   % lower bound to avoid NaN
        
        % build RHS: W*f + beta*Dt(q)
        qx = d_x - b_x;
        qy = d_y - b_y;
        
        rhs = omega .* f; % vectorized elementwise
        % add beta * Dt(q)
        Dtq = applyDt(qx, qy, a, alpha, R);
        rhs = rhs + beta * Dtq;
        % add mu-term to rhs if mu>0 => rhs unaffected, mu multiplies diag in matvec
        
        % define matvec for PCG: (W + beta*DtD + mu*I) * x
        matvec = @(x) matvec_W_beta_DtD(x, omega, beta, mu, a, alpha, R);
        
        % use MATLAB pcg (matrix-free) to solve linear system for u (vector form)
        bvec = rhs(:);
        % initial guess
        x0 = u(:);
        % Use function handle wrapper for pcg
        [x_sol, flag, relres, iter] = pcg(@(v) matvec(reshape(v, Nx, Ny)), bvec, pcgTol, pcgMaxIt, [], [], x0);
        if flag ~= 0 && verbose
            fprintf('  PCG warning: flag=%d relres=%.2e iters=%d\n', flag, relres, iter);
        end
        u = reshape(x_sol, Nx, Ny);
    end
    
    % d-update (shrinkage)
    [Du_x, Du_y] = applyD(u, a, alpha, R);
    qx = Du_x + b_x;
    qy = Du_y + b_y;
    qnorm = sqrt(qx.^2 + qy.^2);
    % avoid division by zero
    shrink = max(0, 1 - (lambda .* wmap) ./ (beta * (qnorm + 1e-12)));
    d_x = shrink .* qx;
    d_y = shrink .* qy;
    
    % b-update
    b_x = b_x + Du_x - d_x;
    b_y = b_y + Du_y - d_y;
    
    % stopping
    relchg = norm(u(:)-prev_u(:))/max(1e-12, norm(prev_u(:)));
    if verbose
        J = compute_energy(u, f, lambda, wmap, a, alpha, R, gamma_const);
        fprintf('Outer %3d: relchg=%.2e Energy=%.6e\n', k, relchg, J);
    end
    if relchg < tolOuter
        if verbose, fprintf('Converged (relchg < tolOuter = %.1e)\n', tolOuter); end
        break;
    end
    prev_u = u;
end

% return
if nargout==0
    imshow(u,[]); title('Restored');
else
    % output variable
end

end

