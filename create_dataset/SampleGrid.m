function [] = SampleGrid
% SAMPLEGRID Generate a equidistantly spaced array in x,y,z around the
% center for the observation point.
% NOTE: This function does not check whether the sampled positions are
% within the defined rectangular room. Hence, no errors will be given if
% the user specifies a microphone array going beyond the walls of the room!

% Import global structure
global Setup

% Create x, y, and z-values for the grid
xValues = (0:Setup.Observation.xSamples-1)*Setup.Observation.xSamplingDistance - (Setup.Observation.xSamples-1)*Setup.Observation.xSamplingDistance/2 + Setup.Observation.Center(1);
yValues = (0:Setup.Observation.ySamples-1)*Setup.Observation.ySamplingDistance - (Setup.Observation.ySamples-1)*Setup.Observation.ySamplingDistance/2 + Setup.Observation.Center(2);
zValues = (0:Setup.Observation.zSamples-1)*Setup.Observation.zSamplingDistance - (Setup.Observation.zSamples-1)*Setup.Observation.zSamplingDistance/2 + Setup.Observation.Center(3);

% Create a sub-struct for each microphone containing the microphone
% x,y,z-coordinates
PointIndex = 1;
for zIndex = 1:length(zValues)
    for yIndex = 1:length(yValues)
        for xIndex = 1:length(xValues)
            Setup.Observation.Point{PointIndex} = [xValues(xIndex), yValues(yIndex), zValues(zIndex)];
            PointIndex = PointIndex + 1;
        end
    end
end

end

