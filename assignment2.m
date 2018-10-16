close all
M = load('flevoland.mat');
R = M.C(1,1,:,:); G = M.C(2,2,:,:); B = M.C(3,3,:,:);
R = squeeze(R); G = squeeze(G); B = squeeze(B);
R = 10*log10(4*pi*R); 
G = 10*log10(4*pi*G); 
B = 10*log10(4*pi*B);
RGB = cat(3,R,G,B);
figure(1);
subplot(1,3,1); imshow(R,[]); title('Red')
subplot(1,3,2); imshow(G,[]); title('Green')
subplot(1,3,3); imshow(B,[]); title('Blue')

figure(2);
imagesc(RGB)
% colormap(M.cmap)