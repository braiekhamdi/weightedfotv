function out = matvec_W_beta_DtD(x_mat, omega, beta, mu, a, alpha, R)
% x_mat is Nx x Ny matrix, return vectorized result of (W + beta D^T D + mu I) * x
% W is diag(omega)
Wx = omega .* x_mat;
% compute D*x
[Dx, Dy] = applyD(x_mat, a, alpha, R);
% compute Dt(Dx,Dy)
DtDx = applyDt(Dx, Dy, a, alpha, R);
out_mat = Wx + beta * DtDx;
if mu>0
    out_mat = out_mat + mu * x_mat;
end
out = out_mat(:);
end
