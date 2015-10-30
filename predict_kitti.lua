#! /usr/bin/env luajit

--[[

This script computes the 3 pixel error on all KITTI 2012 training examples with
the fast architecture.

Don't use this script to fit hyperparameters; the error is computed on the
training examples.

This is not the fastest way to use the neural network---a new process is
spawned and the network is loaded from disk for each image pair---but
is probably the safest.

Usage
-----

   $ ./predict_kitti.lua
   0   0.0028267929719645
   1   0.026568045683624
   2   0.039333925127797
   ...
   191 0.078452818068974
   192 0.012351983422143
   193 0.066736774940625
   0.03222369495401

]]--

require 'image'
require 'torch'
require 'libadcensus'

path = 'data.kitti/unzip'
cmd = './main.lua kitti fast -a predict' ..
   ' -net_fname net/net_kitti_fast_-a_train_all.t7' ..
   ' -left %s -right %s -disp_max 228'

err_sum = 0
n_te = 194
for i = 0, n_te - 1 do
   -- call mc-cnn
   local im0 = ('%s/training/image_0/%06d_10.png'):format(path, i)
   local im1 = ('%s/training/image_1/%06d_10.png'):format(path, i)
   local im = image.loadPNG(im0)
   local img_height = im:size(2)
   local img_width = im:size(3)
   os.execute(cmd:format(im0, im1) .. ' > /dev/null')
   local disp = torch.FloatTensor(torch.FloatStorage('disp.bin')):view(1, 1, img_height, img_width)

   -- ground truth
   local ground_truth = torch.FloatTensor(1, img_height, img_width)
   adcensus.readPNG16(ground_truth, ('%s/training/disp_noc/%06d_10.png'):format(path, i))

   -- compute the error
   local mask = torch.ne(ground_truth, 0):float()
   local bad = torch.add(disp, -1, ground_truth):abs():gt(3):float():cmul(mask)
   local err = bad:sum() / mask:sum()
   err_sum = err_sum + err
   print(i, err)

   collectgarbage()
end
print(err_sum / n_te)
