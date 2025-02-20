function UBP()

clearvars
close all

% load the initial pressure distribution as shepp-Logan phantom
p0_magnitude = 1;

grid_size_SL = 412;
p = p0_magnitude.*phantom('Modified Shepp-Logan',grid_size_SL);
% picture1=imread('./ori.png');
% picture1=im2double(picture1);
% p=picture1;



tot_grid_size = grid_size_SL + 100; %% the extra 100 grid points are used for PML and placing transducers
p0 = zeros(tot_grid_size);

p0(51:tot_grid_size-50,51:tot_grid_size-50) = p;

% assign the grid size and create the computational grid
PML_size = 20; % size of the PML in grid points

Nx = tot_grid_size; % number of grid points in the x direction
Ny = tot_grid_size; % number of grid points in the y direction
x= 10e-3; % total grid size [m]
y = 10e-3; % total grid size [m]
dx = x / Nx; % grid point spacing in the x direction [m]
dy = y / Ny; % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% smooth the initial pressure distribution and restore the magnitude
p0 = smooth(kgrid, p0, true);

% assign to the source structure
source.p0 = p0;

% define the properties of the propagation medium
sound_speed = 1500;
medium.sound_speed = sound_speed; % [m/s]

% define a centered Cartesian circular sensor
sensor_radius = 4.5e-3; % [m]
sensor_angle = 2*pi; % [rad]
sensor_pos = [0, 0]; % [m]
num_sensor_points = 64;
cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);

% assign to sensor structure
sensor.mask = cart_sensor_mask;

% create the time array
[kgrid.t_array, dt]=makeTime(kgrid,medium.sound_speed);

% set the input options
input_args = {'Smooth', false, 'PMLInside', true, 'PlotPML', true};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

%% back projection

t_array = kgrid.t_array;
dt = kgrid.dt;
fs = 1/dt;

np = length(t_array);
NFFT = 2^nextpow2(np);
f_vec = fs*linspace(0,1,NFFT);
w_vec = 2*pi*f_vec;
k_vec = ifftshift(2*pi*f_vec/(sound_speed));

weighting_omega = 0;
p0_recon = zeros(Nx, Ny);

for ii = 1:num_sensor_points

x_ii = cart_sensor_mask(1,ii);
y_ii = cart_sensor_mask(2,ii);

distance_x_square = (kgrid.x - x_ii).^2;
distance_y_square = (kgrid.y - y_ii).^2;
distance_xy = sqrt(distance_x_square + distance_y_square);
distance_xy_index = round(distance_xy/(sound_speed*dt));
distance_xy_index(~distance_xy_index)=1;

p_ii = sensor_data(ii,:);

p_ii_derivative0 = ifft((1i*k_vec).*fft(p_ii,NFFT));
p_ii_derivative = real((p_ii_derivative0(1:np)));

bp_ii = (p_ii - sound_speed.*t_array.*p_ii_derivative);
p0_ii = bp_ii(distance_xy_index);
figure(1)
imagesc((p0_ii))
%%% weighting angle
weighting_arc_ii = (2*pi*sensor_radius)/num_sensor_points;
weighting_phy_ii = 2*pi/num_sensor_points;
weighting_omega_ii = weighting_arc_ii*(1./((distance_x_square + distance_y_square).*sensor_radius).*(-x_ii.*kgrid.x - y_ii.*kgrid.y + sensor_radius.^2));
weighting_omega_ii(isnan(weighting_omega_ii)) = weighting_phy_ii;
weighting_omega_ii(isinf(weighting_omega_ii)) = weighting_phy_ii;

p0_ii = weighting_omega_ii.*p0_ii;

weighting_omega = weighting_omega + weighting_omega_ii;

p0_recon = p0_recon + p0_ii;
figure(2)
imagesc((p0_recon))
end

p0_recon0 = p0_recon./(2*pi);

figure(2) %% reconstructed image
imagesc(kgrid.y_vec*1000,kgrid.x_vec*1000,p0_recon0,[-1,1]) %./max(p0_recon(:))
axis image
colorbar
ylabel('x(mm)');
xlabel('y(mm)');

slice_pos = 4.5e-3; %% comparison with original image amplitude along a horizontal line

figure(2)
plot(kgrid.y_vec,p0(round(slice_pos/kgrid.dx),:),'r')
hold on
plot(kgrid.y_vec,p0_recon0(round(slice_pos/kgrid.dx),:))
hold off
axis tight
xlabel('y(mm)');
ylabel('Reconstructed amplitude');
%% ‘≠Õº
figure;imshow(p);
imwrite(p,'ori.png');
aaa
% 
% %% ÷ÿΩ®
% figure;imshow(p0_recon0,[]);
% imwrite(p0_recon0,'xueguang.png')
% 
% recpnew=p0_recon0(51:1:562,51:1:562);
% % a_max=max(max(recpnew));
% % a_min=min(min(recpnew));
% % recpnew=(recpnew-a_min)/(a_max-a_min);
% imwrite(recpnew,'xueguang1.png');
% imshow(recpnew);
% aaa
end