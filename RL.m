% run the Richardson-Lucy updates

function [x,r,m,t] = RL(Afun, Atfun, b, lb, ub, x, ground, maxIters, bQuiet)

    if nargout>1
        r = nan(maxIters,1);
        m = nan(maxIters,1);
    end
    
    % initial guess
    AtONE = Atfun(ones(size(b)));           
    
    for k=1:maxIters
        tic
        % multiplicative update
        div = b./Afun(x);
        div(isnan(div))=0;
        x = Atfun(div).*x./AtONE ;
                   
        % Clip output
        if ~isempty(lb)            
            x(x<lb) = lb; 
        end
        if ~isempty(ub)
            x(x>ub) = ub;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if (nargout>1) && ~bQuiet
        
            % compute residual
            bb          = (b-Afun(x));                       
            residual    = real(sum(bb(:).^2));
            r(k)        = residual;
            m(k)        = real(mean((ground(:)-x(:)).^2));
            t(k) = toc; 
            % plot current residual
            if ~bQuiet
                disp(['  RL iter ' num2str(k) ' | ' num2str(maxIters) ', residual: ' num2str(residual)]);
            end
        end
    
    end
   
end
