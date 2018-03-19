%% load simulated data
%load sim_data;
%ground = volume;


%% add path for solvers and operators
addpath('scripts/solvers');
addpath('scripts/operators');

psf = load('E:\Users\aphatak\Downloads\psf.mat');
OTF = psf2otf(psf.psf);
stack = load('E:\Users\aphatak\Downloads\stack.mat');
grn = load('E:\Users\aphatak\Downloads\ground.mat');
focalStack = stack.stack;
ground = grn.ground;
%% Set initial guess and parameter
padSize = [0 0 0]; %no padding
% initial guess
x = padarray(focalStack, padSize);

% maximum number of iterations00
maxIters = 100;

%Parameters
K_spectral = 1.005;
rho = 0.25;

% function handles
Afun    = @(x) opAx(x,padSize);
Atfun   = @(x) opAtx(x,padSize);

%% Run Rl, ADMM, CP
% RL 
[x,r,m,t] = RL(Afun, Atfun, focalStack, 0, 1, x, ground, maxIters, false);
figure(); imagesc(max(x,[],3)); colormap gray; axis equal off;
PSNR = 10*log10((max(ground(:)).^2)/m(maxIters));
title(strcat('RL ', num2str(round(PSNR,3))));

% initial guess + ADMM
x = padarray(focalStack, padSize);
[x,r,m,t] = ADMM(rho, focalStack, x, ground, maxIters);
figure(); imagesc(max(real(x),[],3)); colormap gray; axis equal off;
PSNR = 10*log10((max(ground(:)).^2)/m(maxIters));
title(strcat('ADMM ', num2str(round(PSNR,3))));


%initial guess + CP 
x = padarray(focalStack, padSize);
[x,r,m] = CP(K_spectral, focalStack, x, ground, maxIters);
figure(); imagesc(max(x,[],3)); colormap gray; axis equal off;
PSNR = 10*log10((max(ground(:)).^2)/m(maxIters));
title(strcat('CP ', num2str(round(PSNR,3))));

 