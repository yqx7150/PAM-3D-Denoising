pic = imread('02.png');
pic = double(pic);
parallel.gpu.enableCUDAForwardCompatibility(true);
% pic = double(imresize(pic,[612 612]));  %有问题
pic = double(padarray(pic,[50 50]));
sensor1 = forward1(pic);
pic2 = backward1(sensor1);
figure(1);imshow(pic,[]);
figure(2);imshow(sensor1,[]);
figure(3);imshow(pic2,[]);
for i=1:100
%     pic2 = pic2-0.1*gradupcompute(pic2,sensor1);
    pic2 = pic2-0.1*backward1(forward1(pic2) - sensor1);
    figure(100); imshow(pic2,[]);
end
