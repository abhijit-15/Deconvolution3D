%Implements the function A*vol = P*C*vol 
%
%Input:  vol is a 3D volume.
%Output: fs is a 3D focal stack.

function fs = opAx(vol)
    global OTF;        
    fs = ifftn(fftn(vol) .* OTF);
end
