function G = compute_energy(u, f, lambda, wmap, a, alpha, R, gamma)
% compute energy value J(u) to monitor
[Nx, Ny] = size(u);
Gf = sum(sum( log(1 + (f - u).^2 / (gamma^2) ) ));
[Dx, Dy] = applyD(u, a, alpha, R);
Gtv = sum(sum( wmap .* sqrt(Dx.^2 + Dy.^2) ));
G = Gf + lambda * Gtv;
end

