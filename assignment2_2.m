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

%% 2b
%PCA for each pixel? pixel is 4D reduce it?
BW = roipoly(RGB);

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
%%
%apply sliding window of size 5x5
% h = fspecial('average', 5);
% targetFil = imfilter(innerMat,h);
targetFil = slidingWindowAvgSAR(innerMat, 5);
imagesc(mat2gray(cos(targetFil).^-1,[0 pi/2]));


%% the covariance matrix 2a

sX = 1/sqrt(2)*(M.sHV + M.sVH);
%C = calculateC(M.sHH, sX, M.sVV, 5);
% calcuate sigma by retriving th necessary value from the diagonal of
% the array contained in the entries of C
row = size(C,1); col = size(C,2);
sighh = zeros(row,col);
sigx = zeros(row,col);
sigvv = zeros(row,col);
for i = 1 : row
    for j = 1 : col
        sighh(i,j) = 10*log10(4*pi*abs(C{i,j}(1,1)));
        sigx(i,j) = 10*log10(4*pi*abs(C{i,j}(2,2)));
        sigvv(i,j) = 10*log10(4*pi*abs(C{i,j}(3,3)));
    end
end
R = sighh; G = sigx; B = sigvv;
RGB = zeros(row, col, 3);
R(~isfinite(R))=-1;
G(~isfinite(G))=-1;
B(~isfinite(B))=-1;
RGB(:,:,2) = mat2gray(G,[min(min(G)) max(max(G))]);
RGB(:,:,3) = mat2gray(B,[min(min(B)) max(max(B))]);
RGB(:,:,1) = mat2gray(R,[min(min(R)) max(max(R))]);
figure
imshow(RGB)

%% find the coherency matrix 2a
N = 1/sqrt(2) * [1 0 1; 1 0 -1; 0 sqrt(2) 0];
sizC = size(C,2);
T = cell(sizC,sizC);
for i = 1 : sizC
    for j = 1 : sizC
        T{i,j} = N * C{i,j} * N';
    end
end

row = size(T,1); col = size(T,2);
Thh = zeros(row,col);
Tx = zeros(row,col);
Tvv = zeros(row,col);
for i = 1 : row
    for j = 1 : col
        Thh(i,j) = 10*log10(4*pi*abs(T{i,j}(1,1)));
        Tx(i,j) = 10*log10(4*pi*abs(T{i,j}(2,2)));
        Tvv(i,j) = 10*log10(4*pi*abs(T{i,j}(3,3)));
    end
end
R = Thh; G = Tx; B = Tvv;
RGB = zeros(row, col, 3);
R(~isfinite(R))=-1;
G(~isfinite(G))=-1;
B(~isfinite(B))=-1;
RGB(:,:,2) = mat2gray(G,[min(min(G)) max(max(G))]);
RGB(:,:,3) = mat2gray(B,[min(min(B)) max(max(B))]);
RGB(:,:,1) = mat2gray(R,[min(min(R)) max(max(R))]);
figure
imshow(RGB)

%%
[v, l] = eig(R);
