clear;
% a=imread('02.png');
% a=double(a);
% b=forward1(a);
% figure(3);imshow(a,[]);
% figure(30);imshow(b,[]);
% size(a)
% size(b)
% function sensordata1= forward1(pic)
pic = imread('03.png');
pic = double(pic);
parallel.gpu.enableCUDAForwardCompatibility(true);
% pic=double(imresize(pic,[612 612]));
pic = double(padarray(pic,[50 50]));
% check if kWave is on the path
if(~exist('kspaceFirstOrder3D.m', 'file'))
    error('kWave toolbox must be on the path to execute this part of the code')
end
% load struct that contains setting information and subSampling mask
load('/home/liuqg/wgj/diffu1/setting_128mat.mat')
% for an explanation of the options, see kWaveWrapper.m
dataCast    =  'gpuArray-single';
smoothP0    = true;
codeVersion = 'Matlab';
inputArgs   = {'PMLSize', 20, 'DataCast', dataCast, 'Smooth', smoothP0,...
    'kWaveCodeVersion', codeVersion, 'PlotSim', false, 'Output', false};
% define function handles for forward and adjoint operator
A    = @(p0) kWaveWrapper(p0, 'forward', kgrid, medium, sensor, inputArgs{:});
Aadj = @(f)  kWaveWrapper(f,  'adjoint', kgrid, medium, sensor, inputArgs{:});

sensordata1 = A(pic);
pic2 = Aadj(sensordata1);
pic2 = pic2(51:1:562,51:1:562);
figure(3);imshow(pic,[]);
figure(30);imshow(sensordata1,[]);
figure(300);imshow(pic2,[]);
% for i = 1:1000
%     pic2 = double(padarray(pic2,[50 50]));
%     pic2 = pic2 - 0.1 * Aadj( A(pic2) - sensordata1 );
%     pic2 = pic2(51:1:562,51:1:562);
%     figure(300); imshow(pic2,[]);
% %     if i==1
% %         figure(1);imshow(pic2,[]);
% %     end
% %     if i==2
% %         figure(2);imshow(pic2,[]);
% %     end
% %     if i==4
% %         figure(4);imshow(pic2,[]);
% %     end   
% %     if i==6
% %         figure(6);imshow(pic2,[]);
% %     end
% %     if i==8
% %         figure(8);imshow(pic2,[]);
% %     end   
% %     if i==10
% %         figure(10);imshow(pic2,[]);
% %     end  
% %     if i==20
% %         figure(20);imshow(pic2,[]);
% %     end 
% %     if i==30
% %         figure(30);imshow(pic2,[]);
% %     end  
% %         
% %     if i==50
% %         figure(50);imshow(pic2,[]);
% %     end
%     if i==300
%         figure(300);imshow(pic2,[]);
%     end
%     if i==400
%         figure(400);imshow(pic2,[]);
%     end
%     if i==500
%         figure(500);imshow(pic2,[]);
%     end 
%     if i==600
%         figure(600);imshow(pic2,[]);
%     end  
%     if i==700
%         figure(700);imshow(pic2,[]);
%     end
%     if i==800
%         figure(800);imshow(pic2,[]);
%     end
%     if i==900
%         figure(900);imshow(pic2,[]);
%     end 
%     if i==1000
%         figure(1000);imshow(pic2,[]);
%     end
% end