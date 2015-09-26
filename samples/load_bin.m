left = memmapfile('../left.bin', 'Format', 'single').Data;
left = reshape(left, [1 70 370 1226]);
right = memmapfile('../right.bin', 'Format', 'single').Data;
right = reshape(right, [1 70 370 1226]);
disp = memmapfile('../disp.bin', 'Format', 'single').Data;
disp = reshape(right, [1 1 370 1226]);
