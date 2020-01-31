function DrawSetup
% Function for drawing the top view of the simulated room with loudspeaker
% and microphone positions.

global Setup
figure

%% Draw the walls
xRange = [0, Setup.Room.Dim(1)];
yRange = [0, Setup.Room.Dim(2)];
plot(xRange,yRange(1)*[1,1],'k','LineWidth',2);
hold on
plot(xRange,yRange(2)*[1,1],'k','LineWidth',2);
plot(xRange(1)*[1,1],yRange,'k','LineWidth',2);
plot(xRange(2)*[1,1],yRange,'k','LineWidth',2);
ylabel('Length [m]')
xlabel('Width [m]')
xlim(xRange); ylim(yRange)
axis square

%% Draw Source
for i = 1:length(Setup.Source.Position)
    plot(Setup.Source.Position{i}(1), Setup.Source.Position{i}(2),'ro', 'LineWidth',2);
end

%% Draw mics
for i = 1:length(Setup.Observation.Point)
    plot(Setup.Observation.Point{i}(1), Setup.Observation.Point{i}(2),'b.', 'LineWidth',2);
end
end

