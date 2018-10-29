%Function applying an averaging sliding window of size wSize
function speckFiltImg = slidingWindowAvgSAR(SAR, wSize)
h = fspecial('average', wSize);
speckFiltImg = imfilter(SAR,h);
%Downsampling by the window size
speckFiltImg = speckFiltImg(1:wSize:end,1:wSize:end);
end