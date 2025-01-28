% Assume these variables are already defined and populated
% altitude_out, velocity_out, Cb_out

% Prepare your data
x = altitude_out'; % Altitude
y = velocity_out'; % Velocity
z = Cb_out';      % Cb

% Center and scale the data
x_mean = mean(x);
x_std = std(x);
y_mean = mean(y);
y_std = std(y);

x_scaled = (x - x_mean) / x_std;
y_scaled = (y - y_mean) / y_std;

% Combine scaled data into a matrix
data_combined = [x_scaled, y_scaled];

% Remove duplicate rows and adjust corresponding z values
[unique_data, unique_idx, ~] = unique(data_combined, 'rows');
z_unique = z(unique_idx);

% Define polynomial degrees to try
polynomial_degrees = [1, 2, 3, 4, 5]; % Extend this list as needed

% Initialize variables to store results
best_r_squared = -Inf;
best_degree = NaN;
best_fit = [];
best_eqn = '';
best_eqn_python = '';

% Iterate through different polynomial degrees
for degree = polynomial_degrees
    % Construct the polynomial type string
    polynomial = ['poly', num2str(degree), '3'];
    
    try
        % Fit the model
        f = fit([unique_data(:,1), unique_data(:,2)], z_unique, polynomial);
        
        % Compute predictions and residuals
        z_pred = feval(f, [unique_data(:,1), unique_data(:,2)]);
        residuals = z_unique - z_pred;
        
        % Compute R-squared
        SS_res = sum(residuals.^2);
        SS_tot = sum((z_unique - mean(z_unique)).^2);
        R2 = 1 - (SS_res / SS_tot);
        
        % Display R-squared for the current model
        disp(['Fit Type: ', polynomial, ', R-squared: ', num2str(R2)]);
        
        % Update the best model if current model is better
        if R2 > best_r_squared
            best_r_squared = R2;
            best_degree = degree;
            best_fit = f;
            
            % Extract the equation for the best fit
            coeffs = coeffvalues(best_fit);
            terms = {'1', 'altitude', 'velocity', 'altitude^2', 'velocity^2', 'altitude*velocity'};
            best_eqn = 'Cb = ';
            best_eqn_python = 'Cb = ';
            for i = 1:length(coeffs)
                if i <= length(terms)
                    if coeffs(i) ~= 0
                        best_eqn = sprintf('%s %.4f*%s +', best_eqn, coeffs(i), terms{i});
                        best_eqn_python = sprintf('%s %.4f*%s +', best_eqn_python, coeffs(i), terms{i});
                    end
                else
                    % Handle higher-order terms if needed
                    term_idx = i - length(terms);
                    if term_idx == 1
                        best_eqn = sprintf('%s %.4f*altitude^3 +', best_eqn, coeffs(i));
                        best_eqn_python = sprintf('%s %.4f*altitude**3 +', best_eqn_python, coeffs(i));
                    elseif term_idx == 2
                        best_eqn = sprintf('%s %.4f*velocity^3 +', best_eqn, coeffs(i));
                        best_eqn_python = sprintf('%s %.4f*velocity**3 +', best_eqn_python, coeffs(i));
                    end
                end
            end
            best_eqn = best_eqn(1:end-1); % Remove the last '+'
            best_eqn_python = best_eqn_python(1:end-1); % Remove the last '+'
            
            % Print formatted equation for Python
            disp('Python Format:');
            disp(best_eqn_python);
        end
    catch ME
        % Handle fitting errors gracefully
        disp(['Error with fit type: ', polynomial, ', Error: ', ME.message]);
    end
end

% Display the best polynomial degree, R-squared value, and equation
disp(['Best Polynomial Degree: ', num2str(best_degree)]);
disp(['Best R-squared: ', num2str(best_r_squared)]);
disp(['Best Equation: ', best_eqn]);

% Plot the best fit
figure;
plot(best_fit, [unique_data(:,1), unique_data(:,2)], z_unique);
title(['Best Polynomial Fit: poly', num2str(best_degree), '3']);

% Plot residuals for the best fit
z_pred_best = feval(best_fit, [unique_data(:,1), unique_data(:,2)]);
residuals_best = z_unique - z_pred_best;

figure;
scatter3(unique_data(:,1), unique_data(:,2), residuals_best, 'filled');
xlabel('Scaled Altitude');
ylabel('Scaled Velocity');
zlabel('Residuals');
title('Residuals Plot for Best Fit');
