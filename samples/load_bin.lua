require 'torch'

left = torch.FloatTensor(torch.FloatStorage('../left.bin')):view(1, 70, 370, 1226)
right = torch.FloatTensor(torch.FloatStorage('../right.bin')):view(1, 70, 370, 1226)
disp = torch.FloatTensor(torch.FloatStorage('../disp.bin')):view(1, 1, 370, 1226)
