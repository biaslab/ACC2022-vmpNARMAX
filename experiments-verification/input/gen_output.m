function output = gen_output(input, phi, params, orders)

    % Time horizon
    T = length(input);

    % Delay orders (M1 = output, M2 = input, M3 = errors)
    M1 = orders.M1;
    M2 = orders.M2;
    M3 = orders.M3;

    % Parameters
    theta = params.theta;
    tau = params.tau;

    % Observation array
    output = zeros(T,1);
    errors = zeros(T,1);

    for k = 1:T
        
        % Generate noise
        errors(k) = sqrt(inv(tau))*randn(1);
    
        % Output
        if k < (max([M1,M2,M3])+1)
            output(k) = input(k) + errors(k);
        else
            % Update history vectors
            x_kmin1 = output(k-1:-1:k-M1);
            z_kmin1 = input(k-1:-1:k-M2);
            r_kmin1 = errors(k-1:-1:k-M3);
            
            % Compute output
            output(k) = theta'*phi([x_kmin1; input(k); z_kmin1; r_kmin1]) + errors(k);
        end
    end
end
