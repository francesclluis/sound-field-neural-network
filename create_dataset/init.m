%% Script for layout of new structure
function init
global Setup
% ===========================
%       Setup
% ---------------------------
% Sampling frequency [Hz]
Setup.Fs = 1200;

% Impulse response length [s]
Setup.Duration = 1;

% ===========================
%       Ambient
% ---------------------------
% Ambient temperature [deg C]
Setup.Ambient.Temp = 20;

% Ambient pressure [Pa]
Setup.Ambient.Pressure = 1000e2;

% Specific heat capacity of dry air, sea level, 0 deg C
% http://en.wikipedia.org/wiki/Heat_capacity
% Isobaric molar heat capacity [J/mol deg K]
cp = 29.07;
% Isochore molar heat capacity [J/mol deg K]
cv = 20.7643;

% Ratio of specific heats
gamma = cp/cv;

% Ideal gas constant [J/kg deg K]
R = 287;

% Density of air [kg/m3]
Setup.Ambient.rho = Setup.Ambient.Pressure/(R*(Setup.Ambient.Temp + 273.15));

% Speed of sound [m/s]
Setup.Ambient.c = sqrt(gamma*Setup.Ambient.Pressure/Setup.Ambient.rho);

% ===========================
%       Room
% ---------------------------

% Room dimensions [x, y, z] [m]
% ITU-R  BS.1116-3
x = 0;
y = 0;
z = 2.4;
while 20 > x*y || 60 < x*y
    x = 2.83 + (4.87-2.83)*rand(1);
    y = 1.1*x + (4.5*x-9.6-1.1*x)*rand(1);
end

Setup.Room.Dim = [x, y, z]; 

% Reverberation time [s]
Setup.Room.ReverbTime = 0.6; 

Setup.Room.Dim2 = x*y;

% ===========================
%       Source Array
% ---------------------------
% Source lower cutoff frequency
Setup.Source.Highpass = 10; % [Hz]

% Source higher cutoff frequency 
Setup.Source.Lowpass = 500; % [Hz]

% Source position [x,y,z] [m]
sx = Setup.Room.Dim(1)*rand(1);
sy = Setup.Room.Dim(2)*rand(1);
sz = 0;
Setup.Source.Position{1} = [sx, sy, sz];
% Setup.Source.Position{2} = [d*rand(1), Setup.Room.Dim(2)/2 + d*rand(1), d*rand(1)];
% Setup.Source.Position{3} = [d*rand(1), Setup.Room.Dim(2) - d*rand(1), d*rand(1)];
% Setup.Source.Position{4} = [Setup.Room.Dim(1) - d*rand(1), Setup.Room.Dim(2) - d*rand(1), d*rand(1)];
% Setup.Source.Position{5} = [Setup.Room.Dim(1) - d*rand(1), Setup.Room.Dim(2)/2+d*rand(1), d*rand(1)];
% Setup.Source.Position{6} = [Setup.Room.Dim(1) - d*rand(1), d*rand(1), d*rand(1)];
% Setup.Source.Position{7} = [1.89, 4.24-0.5, d*rand(1)];
% Setup.Source.Position{8} = [4.14, 4.35-0.5, d*rand(1)];
Setup.Source.SrcNum = length(Setup.Source.Position);

% ===========================
%       Observation region
% ---------------------------
% Noise in microphones
% Setup.Observation.NoiseLevel = 0; % [dB SPL]

% Observation point x,y,z-position [m]
Setup.Observation.xSamples = 32; 
Setup.Observation.ySamples = 32; 
Setup.Observation.zSamples = 1;
Setup.Observation.xSamplingDistance = Setup.Room.Dim(1)/(Setup.Observation.xSamples-1); % [m]
Setup.Observation.ySamplingDistance = Setup.Room.Dim(2)/(Setup.Observation.ySamples-1); % [m]
Setup.Observation.zSamplingDistance = 1; % [m]

% Center of microphone array [x, y, z] [m]
Setup.Observation.Center = [Setup.Room.Dim(1)/2,Setup.Room.Dim(2)/2,0]; 


SampleGrid





