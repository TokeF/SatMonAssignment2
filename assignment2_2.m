
%% Load data
M = load('sanfransisco.mat');

%% part 2a
R = abs(M.sHH); G = abs(M.sHV); B = abs(M.sVV);
R = 10*log10(4*pi*R); 
G = 10*log10(4*pi*G); 
B = 10*log10(4*pi*B);

% RGB = cat(3,R,G,B);
RGB(:,:,1) = mat2gray(R,[min(min(R(isfinite(R)))) max(max(R))]);
RGB(:,:,2) = mat2gray(G,[min(min(G(isfinite(G)))) max(max(G))]);
RGB(:,:,3) = mat2gray(B,[min(min(B(isfinite(B)))) max(max(B))]);
figure(1); 
subplot(1,3,1); imshow(R,[]); title('Red')
subplot(1,3,2); imshow(G,[]); title('Green')
subplot(1,3,3); imshow(B,[]); title('Blue')

figure(2);
imagesc(RGB)
title('RGB image')

%%
%PCA for each pixel? pixel is 4D reduce it?
% BW = roipoly(RGB);
idx = find(BW == 1);
linsVV = reshape(M.sVV,2800*2800,1); % "one" row of pixels, ie. 3x3xr*c
% [I,J] = ind2sub([2800, 2800], idx);
pix = zeros(length(idx),4);
pix(:,1) = linsVV(idx);
% pix = [M.sVV(I(1),J(1)) M.sHH(I(1),J(1)) M.sHV(I(1),J(1)) M.sVH(I(1),J(1))];
coeff = pca(abs(pix))






