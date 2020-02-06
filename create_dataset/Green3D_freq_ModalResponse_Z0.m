function [Psi_r, Mu, Psi_s] = Green3D_freq_ModalResponse_Z0(FreqLim)
% Green's function for a lightly damped rectangular room.
% Determines the response in the frequency domain for the settings defined
% in the global struct 'Setup'. Assumes uniform distribution of absorption
% and band-pass filters the response. In this file, the frequency response 
% of each mode is returned.
%
% The expressions in this script are based on F. Jacobsen and P. M. Juhl
% "Fundamentals of General Linear Acoustics", Wiley, 2013, ISBN 978-1-118-34641-9
%
% Modification: This file has been modified to only consider eigenfunctions in 2D
% (removing all the height-variation eigenfunctions).
%
% Input parameters:
% ----------------------
% FreqLim:              Scalar
%                       Highest eigenfunction resonance frequency included in the
%                       calculations
%
%
% Output parameters:
% ----------------------
% Psi_s                 Matrix (ndim = 2)
%                       Eigenfunctions evaluated at source positions (size = [nMod,sPos]) 
% Psi_r:                Matrix (ndim = 2)
%                       Eigenfunctions evaluated at receiver positions (size = [rPos,nMod]) 
% sPos:                 Scalar                      
%                       Number of source positions
% rPos:                 Scalar                      
%                       Number of receiver positions
% nFreq:                Scalar
%                       Number of discrete frequency points
% nMod:                 Scalar
%                       Number of eigenfunctions
% Mu:                   Matrix (ndim = 2)
%                       Eigenvalues of the eigenfunctions at each
%                       excitation frequency
%                       (size = [nMod,nFreq])
%
% Author: Martin Møller
% Original version 2017-04-06
% Revision: 

%% Initialize parameters for the room
global Setup

% Determine absorption coefficient from Sabine's equation
V = prod(Setup.Room.Dim);
S = 2*(prod(Setup.Room.Dim(1:2)) + prod(Setup.Room.Dim(2:3)) + prod(Setup.Room.Dim([1,3])));
% Absorption coefficient
alpha = 24*log(10)/Setup.Ambient.c * V/(S*Setup.Room.ReverbTime);
% Normalized specific wall admittance (beta = rho*c*Y)
beta = 1/8*alpha;
% Determine the time-constants of the different eigenfunctions types based 
% on the assumption that the absorption is uniformly distributed and the room 
% is cubic.
tauOblique = V/(2*Setup.Ambient.c*S*beta);
tauTangential = 3*V/(5*Setup.Ambient.c*S*beta);
tauAxial = 3*V/(4*Setup.Ambient.c*S*beta);
tauCompression = V/(Setup.Ambient.c*beta) * 1/S;

% Determine solution frequencies
Frequency = 0:1/(Setup.Duration):Setup.Fs/2-1/(Setup.Duration);
w = 2*pi*Frequency;

% Low frequency rolloff of driver
[B,A] = butter(2,2*Setup.Source.Highpass/Setup.Fs,'high');
Imp = [1,zeros(1,length(w)-1)];
Imp = filter(B,A,Imp);

% High frequency rolloff / anti-aliasing filter
[B,A] = butter(2,2*Setup.Source.Lowpass/Setup.Fs);
Imp = filter(B,A,Imp);
FreqWin = fft(Imp,2*length(w));
FreqWin = FreqWin(1:length(w));

% Initialize x, y, z and fill with the observation point coordinates
x = NaN(length(Setup.Observation.Point),1); y = x; z = x;
for i=1:length(Setup.Observation.Point)
    x(i) = Setup.Observation.Point{i}(1);
    y(i) = Setup.Observation.Point{i}(2);
    z(i) = 0;
end
xS = NaN(1,length(Setup.Source.Position)); yS = xS; zS = xS;
for i=1:length(Setup.Source.Position)
    xS(i) = Setup.Source.Position{i}(1);
    yS(i) = Setup.Source.Position{i}(2);
    zS(i) = 0;
end
k = w/Setup.Ambient.c;


% Frequency limit for modes
MinDim = min(Setup.Room.Dim(1:2));
% Calculate the highest axial modal number with resonance frequency below
% FreqLim (the highest modal number which could be relevant)
MaxModalNumber = ceil(2*FreqLim*MinDim/Setup.Ambient.c);
% Create list of all possible combinations of modal numbers
ModalNumbers = pickn(0:MaxModalNumber,2,'all');
% Calculate resonance frequencies corresponding to the list of modal number
% combinations
ResFreqs = Setup.Ambient.c/(2*pi)* sqrt(sum( (pi* ModalNumbers ./ repmat(Setup.Room.Dim(1:2),size(ModalNumbers,1),1)).^2, 2) );
% Sort modes according to resonance frequency
[~, sortedIdx] = sort(ResFreqs);
ResFreqs = ResFreqs(sortedIdx);
ModalNumbers = ModalNumbers(sortedIdx,:);
% Prune the list of modes to only include modes with resonance frequencies 
% below FreqLim
ModalNumbers = ModalNumbers(ResFreqs<FreqLim,:);
km = 2*pi*ResFreqs(ResFreqs<FreqLim,:)/Setup.Ambient.c;


%% Calculate response
% Initialize transfer function matrix
Psi_s = zeros(length(km),Setup.Source.SrcNum);
Psi_r = zeros(length(Setup.Observation.Point),length(km));
Mu = zeros(length(km),length(w));

for ModeIndex = 1 : size(ModalNumbers,1)
    % eps is the normalization coefficient for the modes
    eps = find(ModalNumbers(ModeIndex)>0);
    % (compression, axial, or tangential)
    if isempty(eps)
        eps = 1;
        len = 0;
    else
        len = length(eps);
        eps = 2^len;
    end
    % Eigenfunctions of the room evaluated at the microphone and source
    % positions. Note that the eigenfunctions have been scaled to be
    % orthonormal.
    Psi_r(:,ModeIndex) = sqrt(eps/V)*...
        cos(ModalNumbers(ModeIndex,1)*pi*x/Setup.Room.Dim(1)) .*...
        cos(ModalNumbers(ModeIndex,2)*pi*y/Setup.Room.Dim(2));
    Psi_s(ModeIndex,:) = sqrt(eps/V)*...
        cos(ModalNumbers(ModeIndex,1)*pi*xS/Setup.Room.Dim(1)) .*...
        cos(ModalNumbers(ModeIndex,2)*pi*yS/Setup.Room.Dim(2)); 
    
    % Apply loss relative to the mode-type
    switch len
        case 0
            % Case compression mode
            taum = tauCompression;
        case 1
            % Case axial mode
            taum = tauAxial;
        case 2
            % Case tangential mode
            taum = tauTangential;
        otherwise
            error('Error in the code');
    end
    % Calculate modal activation according to real-valued, causal time
    % approximation of Green's function (eq. 8.35 in Jacobsen and Juhl)
    % The transfer functions are scaled to correspond to point sources
    % producing 1 Pa amplitude at each excitation frequency in free field 
    % in the pass-band of FreqWin
    Mu(ModeIndex,:) = -4*pi./(k.^2 - km(ModeIndex)^2 - 1j*k/(taum*Setup.Ambient.c)) .* FreqWin;
end
% Hardcode DC-component to zero
Mu(:,1) = 0;

%% Help functions 
function B = pickn(A,N,P)

% B = PICKN(A,N,P)
% 
% Returns P random Picks of N items from vector A.
% 
% PickN assumes that each item can be repelaced after
% selection.
% 
% Example:
%   A list contains four elements: [1 2 3 4]
%   Show two possible combinations of choosing 3 items
%   from the list.
% 
% 
% A = [1:4];
% N = 3;
% P = 2;
% 
% B = pickn(A,N,P)
% 
%       1   3   1
%       2   4   1
% 
% 
% If you specifcy P as the string 'all', then B will contain
% all possible permutations.
%
%
% Richard Medlock, 9-Apr-2003.



% For debug:
% A = [1:4];                  % Input Vector, A
L = length(A);              % Number of Elements in A
% N = 3;                      % Number of Picks (and Number of Columns)
R = L^N;                    % Total Number Of Picks Possible (Rows)
for c = 1:N                 % For Each Column...
    
    cpys = L^(c-1);         % Number of Copies of Each Element in A.
    vsize = cpys*L;         % Size of Vector
    
    for e = 1:cpys          % for each element
        
        counter = 0;        % initialize a counter.
        
        % output index in steps of cpys until the output vector is filled.
        for i = e:cpys:vsize; 

            counter = counter + 1;
            v(i) = A(counter);
            
        end
    end
    
    % Number of repeats of v to fill R...
    M = R/vsize;
    
    % Repeat the matrix...
    B(:,c) = repmat(v',M,1);
    
end

B = fliplr(B);              % Put into a nicer order for viewing.

switch P                    % See how many selections the user wants.
    
    case 'all'              % If they want to see all...
        % don't do anything - just return all of B.                    
        
    otherwise
                            % Generate P random numbers where 1 <= P <= R 
        p = round(1 + (R-1) * rand(P,1));
        B = B(p,:);         % Return those picks.
end
