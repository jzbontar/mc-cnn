left = memmapfile('../left.bin', 'Format', 'single').Data;
left = permute(reshape(left, [1226 370 70]), [3 2 1]);
right = memmapfile('../right.bin', 'Format', 'single').Data;
right = permute(reshape(right, [1226 370 70]), [3 2 1]);
disparity = memmapfile('../disp.bin', 'Format', 'single').Data;
disparity = reshape(disparity, [1226 370])';
