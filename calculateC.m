%Calculate C matrix. Input should be the channel Shh, Sx and Svv
%along with window size wSize used for multiview.
function C = calculateC(Shh, Sx, Svv, wSize)
    %get the multiview of the various channels
    ShhM = slidingWindowAvgSAR(abs(Shh).^2, wSize);
    SvvM = slidingWindowAvgSAR(abs(Svv).^2, wSize);
    SxM = slidingWindowAvgSAR(abs(Sx).^2, wSize);
    ShhxM = slidingWindowAvgSAR(Shh.*conj(Sx), wSize);
    ShhvvM = slidingWindowAvgSAR(Shh.*conj(Svv), wSize);
    SxhhM = slidingWindowAvgSAR(Sx.*conj(Shh), wSize);
    SxvvM = slidingWindowAvgSAR(Sx.*conj(Svv), wSize);
    SvvhhM = slidingWindowAvgSAR(Svv.*conj(Shh), wSize);
    SvvxM = slidingWindowAvgSAR(Svv.*conj(Sx), wSize);
    % C is a cell array. Each entry in C contains a 3x3 double array
    row = size(ShhM,1); col = size(ShhM,2);
    C = cell(row, col);
    %iterate trough all entries in C to calulate Cl for all pixels
    for i = 1 : row
        for j = 1 : col
            cij = [ShhM(i,j) ShhxM(i,j) ShhvvM(i,j);
                   SxhhM(i,j) SxM(i,j) SxvvM(i,j);
                   SvvhhM(i,j) SvvxM(i,j) SvvM(i,j)];
        C{i,j} = cij;
        end
        i
    end
end