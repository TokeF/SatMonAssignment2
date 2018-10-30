%% Load data
M = load('sanfransisco.mat');

%% the covariance matrix 2a
% Calcuate covariance matrix C using function from the previous assignment
sX = 1/sqrt(2)*(M.sHV + M.sVH);
C = calculateC(M.sHH, sX, M.sVV, 5);
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
% stretch and plot covariance matrix
R(~isfinite(R))=-1;
G(~isfinite(G))=-1;
B(~isfinite(B))=-1;
RGB(:,:,2) = mat2gray(G,[min(min(G)) max(max(G))]);
RGB(:,:,3) = mat2gray(B,[min(min(B)) max(max(B))]);
RGB(:,:,1) = mat2gray(R,[min(min(R)) max(max(R))]);
figure
imshow(RGB)

%% find the coherency matrix 2a
% compute coherence matrix from the lienar relation between T and C, using
% the matrix N
N = 1/sqrt(2) * [1 0 1; 1 0 -1; 0 sqrt(2) 0];
sizC = size(C,2);
T = cell(sizC,sizC);
for i = 1 : sizC
    for j = 1 : sizC
        T{i,j} = N * C{i,j} * N';
    end
end

% plot T using the diagonal
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
B = Thh; R = Tx; G = Tvv;
RGB = zeros(row, col, 3);
R(~isfinite(R))=-1;
G(~isfinite(G))=-1;
B(~isfinite(B))=-1;
RGB(:,:,2) = mat2gray(G,[min(min(G)) max(max(G))]);
RGB(:,:,3) = mat2gray(B,[min(min(B)) max(max(B))]);
RGB(:,:,1) = mat2gray(R,[min(min(R)) max(max(R))]);
figure
imshow(RGB)

%% 2b
%find the ROI
BW = roipoly(RGB);

%find the index of the pixels of the ROI
idx = find(BW == 1);
% "one" row of pixels, ie. 3x3xr*c
linsVV = reshape(M.sVV,2800*2800,1);
linsHH = reshape(M.sHH,2800*2800,1);
linsHV = reshape(M.sHV,2800*2800,1);
linsVH = reshape(M.sVH,2800*2800,1);
%store the values of ROI in pix
pix = zeros(length(idx),4);
pix(:,1) = linsVV(idx);
pix(:,2) = linsHH(idx);
pix(:,3) = linsHV(idx);
pix(:,4) = linsVH(idx);
%compute sX and target vector
sX = 1/sqrt(2)*(linsHV + linsVH);
target = [linsHH sX linsVV];
%perform pca on the taget vector of the ROI, each pixels is a measurement
%with 3 variables
coeff = pca(target);
c1 = coeff(:,1);

% compute the image as a taget vector in 2800x2800x3
targetImg(:,:,1) = M.sHH;
targetImg(:,:,2) =  1/sqrt(2)*(M.sHV + M.sVH);
targetImg(:,:,3) = M.sVV;

%Find the angle of each pixel wr. to the first PCA component
innerMat = zeros(2800,2800);
for i = 1 : 2800
    i
    for j = 1 : 2800
        kl = squeeze(targetImg(i,j,:));
        innerMat(i,j) = acos( norm(kl' * c1) / norm(kl));
    end
end
%%
%apply sliding window of size 5x5 and downsampling
targetFil = slidingWindowAvgSAR(innerMat, 5);
imagesc(mat2gray(cos(targetFil).^-1,[0 pi/2]));

%% H and alpha decomposition 2c
% Compute H for each pixel according to the definition
log3 = @(x) log(x) / log(3);
H = zeros(row,col);
alpha = zeros(row,col);
for i = 1: row
    for j = 1 : col
        p = zeros(1,3);
        [v, lam] = eig(T{i,j});
        for k = 1 : 3
            p(k) = lam(1,1) / sum(diag(lam));
        end
        H(i,j) = - (p(1) * log3(p(1)) + p(2) * log3(p(2)) + p(3) * log3(p(3)));
        alpha(i,j) = acos(abs(v(1,1)));
    end
end
alpha = rad2deg(alpha);
%%
Hvec = reshape(abs(H), 560*560, 1);
Avec = reshape(alpha, 560*560, 1);
scatter(Hvec(1:50:313600), Avec(1:50:313600))
xlabel('H values'); ylabel('alpha values')

%% part 2d
close all;
% I don't know how to compute alhpa, so use the angle from part 2b instead
theta = mat2gray(cos(targetFil).^-1,[0 pi/2]);
thetaM = reshape(theta, 560*560,1);

figure
% perform k-means using H and theta as features for each pixel
klbls = kmeans( [Hvec, Avec, thetaM] ,5);
imagesc(reshape(klbls, 560, 560))

figure
% perform k-means using H and theta as features for each pixel
klbls = kmeans( [Hvec, thetaM] ,3);
imagesc(reshape(klbls, 560, 560))

figure
% perform k-means using H and alpha as features for each pixel
klbls = kmeans( [Hvec, Avec] ,3);
imagesc(reshape(klbls, 560, 560))