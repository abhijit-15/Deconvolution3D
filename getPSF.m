clc; clear all;

fname = '36797.tif';
info = imfinfo(fname);
num_images = numel(info);

myscale = [475,476];
fullstack = zeros([myscale,num_images]);
ground = zeros([myscale,num_images/2]);
psf = zeros(size(ground)); 
deconvolved = zeros(size(ground)); 

for k = 1:num_images
    A = im2double(imread(fname, k));
    A = imresize(A , myscale);
    fullstack(:,:,k) = A;
end

stack = fullstack(:,:,1:44);
ground = fullstack(:,:,45:88);

save('stack.mat','stack','-v6');
save('ground.mat','ground','-v6');


for i = 1 : 44
    i
    [J P] = deconvblind(stack(:,:,i) , ones(myscale) , 30);
    psf(: , : , i) = P;
    deconvolved(: , : , i) = J;
end 

save('psf.mat' , 'psf','-v6');
save('deconvolved.mat' , 'deconvolved','-v6');