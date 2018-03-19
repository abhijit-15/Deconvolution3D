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

% maximum number of iterations
maxIters = 100;

% function handles
Afun    = @(x) opAx(x);
Atfun   = @(x) opAtx(x);

% RL 
[x,r,m,t] = RL(Afun, Atfun, focalStack, 0, 1, x, ground, maxIters, false);
figure(); imagesc(max(x,[],3)); colormap gray; axis equal off;
PSNR = 10*log10((max(ground(:)).^2)/m(maxIters));
title(strcat('RL - ', num2str(round(PSNR,3))));
