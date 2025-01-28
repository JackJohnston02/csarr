% Housekeeping
clear all;
close all;

Cb = linspace(1000, 3000, 50);

dt = 0.5;
apogee = 3000;
altitude_out = []; % Initialize array to store altitudes
velocity_out = []; % Initialize array to store velocities
Cb_out = [];       % Initialize array to store ballistic coefficients

tic;

figure_waitbar = waitbar(0,'Please wait...');
for i = 1:length(Cb)
    frac_complete = i/length(Cb); 
    waitbar(frac_complete,figure_waitbar,"Time remaining: " + string(round((toc / frac_complete) * (1 - frac_complete))) + "s");
    x(1) = apogee;
    x(2) = 0;
    x(3) = get_gravity(x(1));
    x(4) = Cb(i);
    
    while x(1) > 1000 
        altitude_out = [altitude_out, x(1)];
        velocity_out = [velocity_out, x(2)];
        Cb_out = [Cb_out, x(4)];

        x = processModel_backward(x, dt);
    end
end

% Scatter plot with color based on Cb_out values
scatter3(altitude_out, velocity_out, Cb_out, 36, Cb_out, 'filled');
colorbar; % Add a colorbar to show the scale of Cb values
xlabel('Altitude');
ylabel('Velocity');
zlabel('Ballistic Coefficient (Cb)');
title('Trajectory colored by Ballistic Coefficient');



function x_new = processModel_backward(x_s, dt)
    dt = -dt;
    g = get_gravity(x_s(1));
    rho = get_density(x_s(1));

    % Update altitude (decreasing since we're propagating backwards)
    x_s(1) = x_s(1) + x_s(2) * dt + 0.5 * x_s(3) * dt^2;
    
    % Update velocity (account for drag and gravity)
    x_s(2) = x_s(2) + x_s(3) * dt;
    
    % Update acceleration (gravity and drag)
    x_s(3) = g - (rho * x_s(2)^2) / (2 * x_s(4));
    
    % Ballistic coefficient remains the same
    x_s(4) = x_s(4);

    x_new = [x_s(1); x_s(2); x_s(3); x_s(4)];
end


function rho = get_density(h)
% Returns atmospheric density as a function of altitude
% Accurate up to 11km
% https://en.wikipedia.org/wiki/Density_of_air

p_0 = 101325; % Standard sea level atmospheric pressure
M = 0.0289652; % molar mass of dry air
R = 8.31445; % ideal gas constant
T_0 = 288.15; % Standard sea level temperature
L = 0.0065; % temperature lapse rate
g = get_gravity(h);

rho = (p_0 * M)/(R * T_0) * (1 - (L * h)/(T_0))^(((-g * M) / (R* L)) - 1); % -g used as g is -ve by default
end

function g = get_gravity(h)
% Returns gravity as a function of altitude
% Approximates the Earth's gravity assumes a perfect sphere

g_0 = -9.80665; % Standard gravity
R_e = 6371000; % Earth radius

g = g_0 * (R_e / (R_e + h))^2;
end
