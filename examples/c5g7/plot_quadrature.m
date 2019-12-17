quadrature = readmatrix('quadrature.csv');
mu = 2 * quadrature(:, 1) - 1;
theta = 2 * pi * quadrature(:, 2);
weight = quadrature(:, 3);
mu_proj = sqrt(1-mu.^2);
polarscatter(theta, mu_proj, 10000*weight, 'filled');