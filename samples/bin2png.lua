require 'cutorch'
require 'image'

d = 70
h = 370
w= 1226

function fromfile(fname)
   local size = io.open(fname):seek('end')
   local x = torch.FloatTensor(torch.FloatStorage(fname, false, size / 4))
   local nan_mask = x:ne(x)
   x[nan_mask] = 1e38
   return x
end

print('Writing left.png')
left = fromfile('left.bin'):view(1, d, h, w)
_, left_ = left:min(2)
image.save('left.png', left_[1]:float():div(d))

print('Writing right.png')
right = fromfile('right.bin'):view(1, d, h, w)
_, right_ = right:min(2)
image.save('right.png', right_[1]:float():div(d))

print('Writing disp.png')
disp = fromfile('disp.bin'):view(1, 1, h, w)
image.save('disp.png', disp[1]:div(d))
