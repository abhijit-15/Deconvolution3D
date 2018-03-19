% Run ADMM for 3D deconvolution

function [x,r,m,t] = ADMM(rho, b, x, ground, maxIters)

    global OTF;
    imageSize = size(b);
    
    p2o = @(x) psf2otf(x, imageSize);

    
    % precompute OTFs 
    cFT     = OTF;
    cTFT    = conj(OTF);

    
    % initialize intermediate variables
    z = zeros([imageSize,2]);
    u = zeros([imageSize,2]);
    x_denominator = (cTFT.*cFT) + 1;
    
    
    % start ADMM iterations
    for iter=1:maxIters    
        tic 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % x update
        v = z-u;
        x_numerator = (cTFT.*p2o(v(:,:,:,1))) + p2o(v(:,:,:,2));
        x = otf2psf(x_numerator./x_denominator);

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % z1 update
        Ax = otf2psf(cFT.*p2o(x));
        v(:,:,:,1) = Ax + u(:,:,:,1);

        t1 = -(1 - rho.*v(:,:,:,1))./(2*rho);
        t21 = (-t1).^2; t22 = b./rho;
        t2 = sqrt(t21+t22);
        z(:,:,:,1) = t1 + t2;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % z2 update
        v(:,:,:,2) = x + u(:,:,:,2);
        z(:,:,:,2) = max(0,v(:,:,:,2));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % u update
        Kx(:,:,:,1) = Ax;
        Kx(:,:,:,2) = x;
        u = u + Kx - z;
        
        % compute residual     
        r(iter) = real(sum( (b(:)-Ax(:)).^2));
        m(iter) = real(mean((ground(:)-x(:)).^2));
        t(iter) = toc;
        
        
        % display status
        disp(['  ADMM iter ' num2str(iter) ' | ' num2str(maxIters) ', residual: ' num2str(r(iter))]);
        
end
