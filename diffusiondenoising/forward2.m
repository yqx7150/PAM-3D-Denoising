function sensordata1= forward2(pic)
parallel.gpu.enableCUDAForwardCompatibility(true);

% pic=double(imresize(pic,[612 612]));
 %pic = double(padarray(pic,[50 50]));
% % 
 pic = double(padarray(pic,[92 92]));
%  Script for calling the kWave operattor for learned 3D photoacoustic
%  imaging. This script initialises the model in k-wave and
%  evaluates the gradients.
% 
%  This is accompanying code for: Hauptmann et al., Model based learning for 
%  accelerated, limited-view 3D photoacoustic tomography, 
%  https://arxiv.org/abs/1708.09832
% 
%  written by Felix Lucka and Andreas Hauptmann, January 2018
%  ==============================================================================

%This part is called as initialisation 
% if(compInit)

    % check if kWave is on the path
    if(~exist('kspaceFirstOrder3D.m', 'file'))
       error('kWave toolbox must be on the path to execute this part of the code') 
    end
    
    % load struct that contains setting information and subSampling mask
     load('/home/liuqg/wgj/diffu4/st_32.mat')

      %load('/home/liuqg/wgj/diffu1/st_1000.mat')
%     recSize = [kgrid.Nx, kgrid.Ny, kgrid.Nz];

    % for these settings, see kWave documentation. 
%     medium               = [];
%     medium.sound_speed   = 1580; % [m/s] speed of sound
%     sensor               = [];      
%     sensor.mask          = false([kgrid.Nx, kgrid.Ny]);
%   sensor.mask(1, :, :) = subSamplingMask;

    % for an explanation of the options, see kWaveWrapper.m
    dataCast    =  'gpuArray-single';
    smoothP0    = true;
    codeVersion = 'Matlab'; 

    inputArgs   = {'PMLSize', 20, 'DataCast', dataCast, 'Smooth', smoothP0,...
        'kWaveCodeVersion', codeVersion, 'PlotSim', false, 'Output', false};
    
    % define function handles for forward and adjoint operator
    A    = @(p0) kWaveWrapper(p0, 'forward', kgrid, medium, sensor, inputArgs{:});
    Aadj = @(f)  kWaveWrapper(f,  'adjoint', kgrid, medium, sensor, inputArgs{:});
    sensordata1=A(pic);
%     pic=Aadj(sensor_data1);
    
   
    
end
