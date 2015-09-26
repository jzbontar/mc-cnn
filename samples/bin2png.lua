require 'cutorch'
require 'image'

print('Writing left.png')
left = torch.FloatTensor(torch.FloatStorage('left.bin')):view(1, 70, 370, 1226):cuda()
_, left_ = left:min(2)
image.save('left.png', left_[1]:float():div(70))

print('Writing right.png')
right = torch.FloatTensor(torch.FloatStorage('right.bin')):view(1, 70, 370, 1226):cuda()
_, right_ = right:min(2)
image.save('right.png', right_[1]:float():div(70))

print('Writing disp.png')
disp = torch.FloatTensor(torch.FloatStorage('disp.bin')):view(1, 1, 370, 1226)
image.save('disp.png', disp[1]:div(70))
