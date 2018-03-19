%Implements the function A'*x = C'*P'*x
% 
%Input:  fs is a 3D focal stack
%Output: vol is a 3D volume

function vol = opAtx(fs)
    global OTF;    
    vol = ifftn(fftn(fs) .* conj(OTF));    
end
