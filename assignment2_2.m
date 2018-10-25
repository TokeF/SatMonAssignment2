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
%BW = roipoly(RGB);

%find the index of the pixels of the ROI
idx = find(BW == 1);
% "one" row of pixels, ie. 3x3xr*c
linsVV = reshape(M.sVV,2800*2800,1);
linsHH = reshape(M.sHH,2800*2800,1);
linsHV = reshape(M.sHV,2800*2800,1);
linsVH = reshape(M.sVH,2800*2800,1);
pix = zeros(length(idx),4);
pix(:,1) = linsVV(idx);
pix(:,2) = linsHH(idx);
pix(:,3) = linsHV(idx);
pix(:,4) = linsVH(idx);
sX = 1/sqrt(2)*(linsHV + linsVH);
target = [linsHH sX linsVV];
%perform pca on the taget vector of the ROI, each pixels is a measurement with 4
%variables
coeff = pca(target);
c1 = coeff(:,1);

% compute the image as a taget vector in 2800x2800x3
targetImg(:,:,1) = M.sHH;
targetImg(:,:,2) =  1/sqrt(2)*(M.sHV + M.sVH);
targetImg(:,:,3) = M.sVV;

innerMat = zeros(2800,2800);
for i = 1 : 2800
    i
    for j = 1 : 2800
        kl = squeeze(targetImg(i,j,:));
        innerMat(i,j) = acos( norm(kl' * c1) / norm(kl));
    end
end

%apply sliding window of size 5x5
h = fspecial('average', 5);
targetFil = imfilter(innerMat,h);

imagesc(mat2gray(cos(targetFil).^-1,[0 pi/2]))

% normkl = sqrt( targetImg(:,:,1) .* conj(targetImg(:,:,1)) ...
%              + targetImg(:,:,2) .* conj(targetImg(:,:,2)) ... 
%              + targetImg(:,:,3) .* conj(targetImg(:,:,3)) );
% 
% %multiply target vectors with c1
% targetImg(:,:,1) = targetImg(:,:,1) * c1(1);
% targetImg(:,:,2) = targetImg(:,:,2) * c1(2);
% targetImg(:,:,3) = targetImg(:,:,3) * c1(3);
% dummyMat = norm(sum(targetImg,3)) ./ normkl;
% 
% %apply sliding window of size 5x5
% h = fspecial('average', 5);
% targetFil = imfilter(dummyMat,h);
% 
% imagesc(mat2gray(cos(targetFil).^-1,[0 pi/2]))

%%
% Transformation matrix U
% U = 1/sqrt(2) * [1 0 1; 1 0 -1; 0 sqrt(2) 0];
% 
% if ~exist('kp')
%     kp = zeros(2800,2800,3);
%     for i = 1 : 2800
%         i
%         for j = 1 : 2800
%             kl = squeeze(targetImg(i,j,:));
%             kp(i,j,:) = U*kl;
%         end
%     end
% end
% 
% %apply sliding window of size 5x5
% h = fspecial('average', 5);
% % coherency matrix T
% T = imfilter(sum(kp.*conj(kp),3),h);
% kpkp = cell(2800,2800);
% for i = 1 : 2800
%     i
%     for j = 1 : 2800
%         kpl = squeeze(kp(i,j,:));
%         kpkp{i,j} = kpl * conj(kpl)';
%     end
% end

T = calcT(M);
