%Load dataset
psf = load('psf.mat');
OTF = psf2otf(psf.psf);
stack = load('stack.mat');
grn = load('ground.mat');
focalStack = stack.stack;
ground = grn.ground;
%-------------------------

x = focalStack;
global OTF;

% maximum number of iterations and rho
maxIters = 100;
rho = 0.21;

% function handles
Afun    = @(x) opAx(x);
Atfun   = @(x) opAtx(x);

%ADMM
[x,r,m,t] = ADMM(rho, focalStack, x, ground, maxIters);
figure(); imagesc(max(real(x),[],3)); colormap gray; axis equal off;
PSNR = 10*log10((max(ground(:)).^2)/m(maxIters));
title(strcat('ADMM ', num2str(round(PSNR,3))));

