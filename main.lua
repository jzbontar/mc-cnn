#! /usr/bin/env luajit

require 'torch'

io.stdout:setvbuf('no')
for i = 1,#arg do
   io.write(arg[i] .. ' ')
end
io.write('\n')
dataset = table.remove(arg, 1)
arch = table.remove(arg, 1)
assert(dataset == 'kitti' or dataset == 'kitti2015' or dataset == 'mb')
assert(arch == 'fast' or arch == 'slow' or arch == 'ad' or arch == 'census')

cmd = torch.CmdLine()
cmd:option('-gpu', 1, 'gpu id')
cmd:option('-seed', 42, 'random seed')
cmd:option('-debug', false)
cmd:option('-d', 'kitti | mb')
cmd:option('-a', 'train_tr | train_all | test_te | test_all | submit | time | predict', 'train')
cmd:option('-net_fname', '')
cmd:option('-make_cache', false)
cmd:option('-use_cache', false)
cmd:option('-print_args', false)
cmd:option('-sm_terminate', '', 'terminate the stereo method after this step')
cmd:option('-sm_skip', '', 'which part of the stereo method to skip')
cmd:option('-tiny', false)
cmd:option('-subset', 1)

cmd:option('-left', '')
cmd:option('-right', '')
cmd:option('-disp_max', '')

if dataset == 'kitti' or dataset == 'kitti2015' then
   cmd:option('-hflip', 0)
   cmd:option('-vflip', 0)
   cmd:option('-rotate', 7)
   cmd:option('-hscale', 0.9)
   cmd:option('-scale', 1)
   cmd:option('-trans', 0)
   cmd:option('-hshear', 0.1)
   cmd:option('-brightness', 0.7)
   cmd:option('-contrast', 1.3)
   cmd:option('-d_vtrans', 0)
   cmd:option('-d_rotate', 0)
   cmd:option('-d_hscale', 1)
   cmd:option('-d_hshear', 0)
   cmd:option('-d_brightness', 0.3)
   cmd:option('-d_contrast', 1)
elseif dataset == 'mb' then
   cmd:option('-hflip', 0)
   cmd:option('-vflip', 0)
   cmd:option('-rotate', 28)
   cmd:option('-hscale', 0.8)
   cmd:option('-scale', 0.8)
   cmd:option('-trans', 0)
   cmd:option('-hshear', 0.1)
   cmd:option('-brightness', 1.3)
   cmd:option('-contrast', 1.1)
   cmd:option('-d_vtrans', 1)
   cmd:option('-d_rotate', 3)
   cmd:option('-d_hscale', 0.9)
   cmd:option('-d_hshear', 0.3)
   cmd:option('-d_brightness', 0.7)
   cmd:option('-d_contrast', 1.1)
end

cmd:option('-rect', 'imperfect')
cmd:option('-color', 'gray')
if arch == 'slow' then
   if dataset == 'kitti' or dataset == 'kitti2015' then
      cmd:option('-at', 0)

      cmd:option('-l1', 4)
      cmd:option('-fm', 112)
      cmd:option('-ks', 3)
      cmd:option('-l2', 4)
      cmd:option('-nh2', 384)
      cmd:option('-lr', 0.003)
      cmd:option('-bs', 128)
      cmd:option('-mom', 0.9)
      cmd:option('-true1', 1)
      cmd:option('-false1', 4)
      cmd:option('-false2', 10)
   
      if dataset == 'kitti' then
         cmd:option('-L1', 5)
         cmd:option('-cbca_i1', 2)
         cmd:option('-cbca_i2', 0)
         cmd:option('-tau1', 0.13)
         cmd:option('-pi1', 1.32)
         cmd:option('-pi2', 24.25)
         cmd:option('-sgm_i', 1)
         cmd:option('-sgm_q1', 3)
         cmd:option('-sgm_q2', 2)
         cmd:option('-alpha1', 2)
         cmd:option('-tau_so', 0.08)
         cmd:option('-blur_sigma', 5.99)
         cmd:option('-blur_t', 6)
      elseif dataset == 'kitti2015' then
         cmd:option('-L1', 5)
         cmd:option('-cbca_i1', 2)
         cmd:option('-cbca_i2', 4)
         cmd:option('-tau1', 0.03)
         cmd:option('-pi1', 2.3)
         cmd:option('-pi2', 24.25)
         cmd:option('-sgm_i', 1)
         cmd:option('-sgm_q1', 3)
         cmd:option('-sgm_q2', 2)
         cmd:option('-alpha1', 1.75)
         cmd:option('-tau_so', 0.08)
         cmd:option('-blur_sigma', 5.99)
         cmd:option('-blur_t', 5)
      end
   elseif dataset == 'mb' then
      cmd:option('-ds', 2001)
      cmd:option('-d_exp', 0.2)
      cmd:option('-d_light', 0.2)

      cmd:option('-l1', 5)
      cmd:option('-fm', 112)
      cmd:option('-ks', 3)
      cmd:option('-l2', 3)
      cmd:option('-nh2', 384)
      cmd:option('-lr', 0.003)
      cmd:option('-bs', 128)
      cmd:option('-mom', 0.9)
      cmd:option('-true1', 0.5)
      cmd:option('-false1', 1.5)
      cmd:option('-false2', 18)

      cmd:option('-L1', 14)
      cmd:option('-tau1', 0.02)
      cmd:option('-cbca_i1', 2)
      cmd:option('-cbca_i2', 16)
      cmd:option('-pi1', 1.3)
      cmd:option('-pi2', 13.9)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 4.5)
      cmd:option('-sgm_q2', 2)
      cmd:option('-alpha1', 2.75)
      cmd:option('-tau_so', 0.13)
      cmd:option('-blur_sigma', 1.67)
      cmd:option('-blur_t', 2)
   end
elseif arch == 'census' then
   if dataset == 'kitti' or dataset == 'kitti2015' then
      cmd:option('-L1', 0)
      cmd:option('-cbca_i1', 4)
      cmd:option('-cbca_i2', 8)
      cmd:option('-tau1', 0.01)
      cmd:option('-pi1', 4)
      cmd:option('-pi2', 128.00)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 3)
      cmd:option('-sgm_q2', 3.5)
      cmd:option('-alpha1', 1.25)
      cmd:option('-tau_so', 1.0)
      cmd:option('-blur_sigma', 7.74)
      cmd:option('-blur_t', 6)
   elseif dataset == 'mb' then
      cmd:option('-L1', 5)
      cmd:option('-cbca_i1', 8)
      cmd:option('-cbca_i2', 8)
      cmd:option('-tau1', 0.22)
      cmd:option('-pi1', 4.0)
      cmd:option('-pi2', 32.0)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 4)
      cmd:option('-sgm_q2', 3)
      cmd:option('-alpha1', 1.5)
      cmd:option('-tau_so', 1.00)
      cmd:option('-blur_sigma', 2.78)
      cmd:option('-blur_t', 3)
   end
elseif arch == 'ad' then
   if dataset == 'kitti' or dataset == 'kitti2015' then
      cmd:option('-L1', 3)
      cmd:option('-cbca_i1', 0)
      cmd:option('-cbca_i2', 4)
      cmd:option('-tau1', 0.03)
      cmd:option('-pi1', 0.76)
      cmd:option('-pi2', 13.93)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 3.5)
      cmd:option('-sgm_q2', 2)
      cmd:option('-alpha1', 2.5)
      cmd:option('-tau_so', 0.01)
      cmd:option('-blur_sigma', 7.74)
      cmd:option('-blur_t', 6)
   elseif dataset == 'mb' then
      cmd:option('-L1', 5)
      cmd:option('-cbca_i1', 0)
      cmd:option('-cbca_i2', 4)
      cmd:option('-tau1', 0.36)
      cmd:option('-pi1', 0.4)
      cmd:option('-pi2', 8.0)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 3)
      cmd:option('-sgm_q2', 4)
      cmd:option('-alpha1', 2.5)
      cmd:option('-tau_so', 0.08)
      cmd:option('-blur_sigma', 7.74)
      cmd:option('-blur_t', 1)
   end
elseif arch == 'fast' then
   if dataset == 'kitti' then
      cmd:option('-at', 0)
      cmd:option('-m', 0.2, 'margin')
      cmd:option('-pow', 1)

      cmd:option('-l1', 4)
      cmd:option('-fm', 64)
      cmd:option('-ks', 3)
      cmd:option('-lr', 0.002)
      cmd:option('-bs', 128)
      cmd:option('-mom', 0.9)
      cmd:option('-true1', 1)
      cmd:option('-false1', 4)
      cmd:option('-false2', 10)

      cmd:option('-L1', 0)
      cmd:option('-cbca_i1', 0)
      cmd:option('-cbca_i2', 0)
      cmd:option('-tau1', 0)
      cmd:option('-pi1', 4)
      cmd:option('-pi2', 55.72)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 3)
      cmd:option('-sgm_q2', 2.5)
      cmd:option('-alpha1', 1.5)
      cmd:option('-tau_so', 0.02)
      cmd:option('-blur_sigma', 7.74)
      cmd:option('-blur_t', 5)
   elseif dataset == 'kitti2015' then
      cmd:option('-at', 0)
      cmd:option('-m', 0.2, 'margin')
      cmd:option('-pow', 1)

      cmd:option('-l1', 4)
      cmd:option('-fm', 64)
      cmd:option('-ks', 3)
      cmd:option('-lr', 0.002)
      cmd:option('-bs', 128)
      cmd:option('-mom', 0.9)
      cmd:option('-true1', 1)
      cmd:option('-false1', 4)
      cmd:option('-false2', 10)

      cmd:option('-L1', 0)
      cmd:option('-cbca_i1', 0)
      cmd:option('-cbca_i2', 0)
      cmd:option('-tau1', 0)
      cmd:option('-pi1', 2.3)
      cmd:option('-pi2', 18.38)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 3)
      cmd:option('-sgm_q2', 2)
      cmd:option('-alpha1', 1.25)
      cmd:option('-tau_so', 0.08)
      cmd:option('-blur_sigma', 4.64)
      cmd:option('-blur_t', 5)
   elseif dataset == 'mb' then
      cmd:option('-m', 0.2, 'margin')
      cmd:option('-pow', 1)

      cmd:option('-ds', 2001)
      cmd:option('-d_exp', 0.2)
      cmd:option('-d_light', 0.2)

      cmd:option('-l1', 5)
      cmd:option('-fm', 64)
      cmd:option('-ks', 3)
      cmd:option('-lr', 0.002)
      cmd:option('-bs', 128)
      cmd:option('-mom', 0.9)
      cmd:option('-true1', 0.5)
      cmd:option('-false1', 1.5)
      cmd:option('-false2', 6)

      cmd:option('-L1', 0)
      cmd:option('-tau1', 0.0)
      cmd:option('-cbca_i1', 0)
      cmd:option('-cbca_i2', 0)
      cmd:option('-pi1', 2.3)
      cmd:option('-pi2', 24.3)
      cmd:option('-sgm_i', 1)
      cmd:option('-sgm_q1', 4)
      cmd:option('-sgm_q2', 2)
      cmd:option('-alpha1', 1.5)
      cmd:option('-tau_so', 0.08)
      cmd:option('-blur_sigma', 6)
      cmd:option('-blur_t', 2)
   end
end

opt = cmd:parse(arg)

if opt.print_args then
   print((opt.ks - 1) * opt.l1 + 1, 'arch_patch_size')
   print(opt.l1, 'arch1_num_layers')
   print(opt.fm, 'arch1_num_feature_maps')
   print(opt.ks, 'arch1_kernel_size')
   print(opt.l2, 'arch2_num_layers')
   print(opt.nh2, 'arch2_num_units_2')
   print(opt.false1, 'dataset_neg_low')
   print(opt.false2, 'dataset_neg_high')
   print(opt.true1, 'dataset_pos_low')
   print(opt.tau1, 'cbca_intensity')
   print(opt.L1, 'cbca_distance')
   print(opt.cbca_i1, 'cbca_num_iterations_1')
   print(opt.cbca_i2, 'cbca_num_iterations_2')
   print(opt.pi1, 'sgm_P1')
   print(opt.pi1 * opt.pi2, 'sgm_P2')
   print(opt.sgm_q1, 'sgm_Q1')
   print(opt.sgm_q1 * opt.sgm_q2, 'sgm_Q2')
   print(opt.alpha1, 'sgm_V')
   print(opt.tau_so, 'sgm_intensity')
   print(opt.blur_sigma, 'blur_sigma')
   print(opt.blur_t, 'blur_threshold')
   os.exit()
end

require 'cunn'
require 'cutorch'
require 'image'
require 'libadcensus'
require 'libcv'
require 'cudnn'
cudnn.benchmark = true

include('Margin2.lua')
include('Normalize2.lua')
include('BCECriterion2.lua')
include('StereoJoin.lua')
include('StereoJoin1.lua')
include('SpatialConvolution1_fw.lua')
-- include('SpatialLogSoftMax.lua')

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))

cmd_str = dataset .. '_' .. arch
for i = 1,#arg do
   cmd_str = cmd_str .. '_' .. arg[i]
end

function isnan(n)
   return tostring(n) == tostring(0/0)
end

function fromfile(fname)
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   if #dim == 1 and dim[1] == 0 then
      return torch.Tensor()
   end

   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   if type == 'float32' then
      x = torch.FloatTensor(torch.FloatStorage(fname))
   elseif type == 'int32' then
      x = torch.IntTensor(torch.IntStorage(fname))
   elseif type == 'int64' then
      x = torch.LongTensor(torch.LongStorage(fname))
   else
      print(fname, type)
      assert(false)
   end

   x = x:reshape(torch.LongStorage(dim))
   return x
end

function get_window_size(net)
   ws = 1
   for i = 1,#net.modules do
      local module = net:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' then
         ws = ws + module.kW - 1
      end
   end
   return ws
end

-- load training data
if dataset == 'kitti' or dataset == 'kitti2015' then
   height = 350
   width = 1242
   disp_max = 228
   n_te = dataset == 'kitti' and 195 or 200
   n_input_plane = 1
   err_at = 3

   if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'test_te' or opt.a == 'test_all' or opt.a == 'submit' then
      if opt.at == 1 then
         function load(fname)
            local X_12 = fromfile('data.kitti/' .. fname)
            local X_15 = fromfile('data.kitti2015/' .. fname)
            local X = torch.cat(X_12[{{1,194}}], X_15[{{1,200}}], 1)
            X = torch.cat(X, dataset == 'kitti' and X_12[{{195,389}}] or X_15[{{200,400}}], 1)
            return X
         end
         X0 = load('x0.bin')
         X1 = load('x1.bin')
         metadata = load('metadata.bin')

         dispnoc = torch.cat(fromfile('data.kitti/dispnoc.bin'), fromfile('data.kitti2015/dispnoc.bin'), 1)
         tr = torch.cat(fromfile('data.kitti/tr.bin'), fromfile('data.kitti2015/tr.bin'):add(194))
         te = dataset == 'kitti' and fromfile('data.kitti/te.bin') or fromfile('data.kitti2015/te.bin'):add(194)

         function load_nnz(fname)
            local X_12 = fromfile('data.kitti/' .. fname)
            local X_15 = fromfile('data.kitti2015/' .. fname)
            X_15[{{},1}]:add(194)
            return torch.cat(X_12, X_15, 1)
         end
         nnz_tr = load_nnz('nnz_tr.bin')
         nnz_te = load_nnz('nnz_te.bin')
      elseif dataset == 'kitti' then
         X0 = fromfile('data.kitti/x0.bin')
         X1 = fromfile('data.kitti/x1.bin')
         dispnoc = fromfile('data.kitti/dispnoc.bin')
         metadata = fromfile('data.kitti/metadata.bin')
         tr = fromfile('data.kitti/tr.bin')
         te = fromfile('data.kitti/te.bin')
         nnz_tr = fromfile('data.kitti/nnz_tr.bin')
         nnz_te = fromfile('data.kitti/nnz_te.bin')
      elseif dataset == 'kitti2015' then
         X0 = fromfile('data.kitti2015/x0.bin')
         X1 = fromfile('data.kitti2015/x1.bin')
         dispnoc = fromfile('data.kitti2015/dispnoc.bin')
         metadata = fromfile('data.kitti2015/metadata.bin')
         tr = fromfile('data.kitti2015/tr.bin')
         te = fromfile('data.kitti2015/te.bin')
         nnz_tr = fromfile('data.kitti2015/nnz_tr.bin')
         nnz_te = fromfile('data.kitti2015/nnz_te.bin')
      end
   end
elseif dataset == 'mb' then
   if opt.color == 'rgb' then
      n_input_plane = 3
   else
      n_input_plane = 1
   end
   err_at = 1

   if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'test_te' or opt.a == 'test_all' or opt.a == 'submit' then
      data_dir = ('data.mb.%s_%s'):format(opt.rect, opt.color)
      te = fromfile(('%s/te.bin'):format(data_dir))
      metadata = fromfile(('%s/meta.bin'):format(data_dir))
      nnz_tr = fromfile(('%s/nnz_tr.bin'):format(data_dir))
      nnz_te = fromfile(('%s/nnz_te.bin'):format(data_dir))
      fname_submit = {}
      for line in io.open(('%s/fname_submit.txt'):format(data_dir), 'r'):lines() do
         table.insert(fname_submit, line)
      end
      X = {}
      dispnoc = {}
      height = 1500
      width = 1000
      for n = 1,metadata:size(1) do
         local XX = {}
         light = 1
         while true do
            fname = ('%s/x_%d_%d.bin'):format(data_dir, n, light)
            if not paths.filep(fname) then
               break
            end
            table.insert(XX, fromfile(fname))
            light = light + 1
            if opt.a == 'test_te' or opt.a == 'submit' then
               break  -- we don't need to load training data
            end
         end
         table.insert(X, XX)

         fname = ('%s/dispnoc%d.bin'):format(data_dir, n)
         if paths.filep(fname) then
            table.insert(dispnoc, fromfile(fname))
         end
      end
   end
end

function savePNG(fname, x, isvol)
   local pred
   local pred_jet = torch.Tensor(1, 3, x:size(3), x:size(4))

   if isvol == true then
      pred = torch.CudaTensor(1, 1, x:size(3), x:size(4))
      adcensus.spatial_argmin(x, pred)
   else
      pred = x:double():add(1)
   end
   adcensus.grey2jet(pred[{1,1}]:div(disp_max):double(), pred_jet)
   image.savePNG(fname, pred_jet[1])
end

function saveOutlier(fname, x0, outlier)
   local img = torch.Tensor(1,3,height,img_width)
   img[{1,1}]:copy(x0)
   img[{1,2}]:copy(x0)
   img[{1,3}]:copy(x0)
   for i=1,height do
      for j=1,img_width do
         if outlier[{1,1,i,j}] == 1 then
            img[{1,1,i,j}] = 0
            img[{1,2,i,j}] = 1
            img[{1,3,i,j}] = 0
         elseif outlier[{1,1,i,j}] == 2 then
            img[{1,1,i,j}] = 1
            img[{1,2,i,j}] = 0
            img[{1,3,i,j}] = 0
         end
      end
   end
   image.savePNG(fname, img[1])
end

function gaussian(sigma)
   local kr = math.ceil(sigma * 3)
   local ks = kr * 2 + 1
   local k = torch.Tensor(ks, ks)
   for i = 1, ks do
      for j = 1, ks do
         local y = (i - 1) - kr
         local x = (j - 1) - kr
         k[{i,j}] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
      end
   end
   return k
end

function print_net(net)
   local s
   local t = torch.typename(net) 
   if t == 'cudnn.SpatialConvolution' then
      print(('conv(in=%d, out=%d, k=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW))
   elseif t == 'nn.SpatialConvolutionMM_dsparse' then
      print(('conv_dsparse(in=%d, out=%d, k=%d, s=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW, net.sW))
   elseif t == 'cudnn.SpatialMaxPooling' then
      print(('max_pool(k=%d, d=%d)'):format(net.kW, net.dW))
   elseif t == 'nn.StereoJoin' then
      print(('StereoJoin(%d)'):format(net.disp_max))
   elseif t == 'nn.Margin2' then
      print(('Margin2(margin=%f, pow=%d)'):format(opt.m, opt.pow))
   elseif t == 'nn.GHCriterion' then
      print(('GHCriterion(m_pos=%f, m_neg=%f, pow=%d)'):format(opt.m_pos, opt.m_neg, opt.pow))
   elseif t == 'nn.Sequential' then
      for i = 1,#net.modules do
         print_net(net.modules[i])
      end
   else
      print(net)
   end
end

function clean_net(net)
   net.output = torch.CudaTensor()
   net.gradInput = nil
   net.weight_v = nil
   net.bias_v = nil
   net.gradWeight = nil
   net.gradBias = nil
   net.iDesc = nil
   net.oDesc = nil
   net.finput = torch.CudaTensor()
   net.fgradInput = torch.CudaTensor()
   net.tmp_in = torch.CudaTensor()
   net.tmp_out = torch.CudaTensor()
   if net.modules then
      for _, module in ipairs(net.modules) do
         clean_net(module)
      end
   end
   return net
end

function save_net(epoch)
   if arch == 'slow' then
      obj = {clean_net(net_te), clean_net(net_te2), opt}
   elseif arch == 'fast' then
      obj = {clean_net(net_te), opt}
   end
   if epoch == 0 then
      fname = ('net/net_%s.t7'):format(cmd_str)
   else
      fname = ('net/net_%s_%d.t7'):format(cmd_str, epoch)
   end
   torch.save(fname, obj, 'ascii')
   return fname
end

if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'time' then
   function mul32(a,b)
      return {a[1]*b[1]+a[2]*b[4], a[1]*b[2]+a[2]*b[5], a[1]*b[3]+a[2]*b[6]+a[3], a[4]*b[1]+a[5]*b[4], a[4]*b[2]+a[5]*b[5], a[4]*b[3]+a[5]*b[6]+a[6]}
   end

   function make_patch(src, dst, dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
      local m = {1, 0, -dim4, 0, 1, -dim3}
      m = mul32({1, 0, trans[1], 0, 1, trans[2]}, m) -- translate
      m = mul32({scale[1], 0, 0, 0, scale[2], 0}, m) -- scale
      local c = math.cos(phi)
      local s = math.sin(phi)
      m = mul32({c, s, 0, -s, c, 0}, m) -- rotate
      m = mul32({1, hshear, 0, 0, 1, 0}, m) -- shear
      m = mul32({1, 0, (ws - 1) / 2, 0, 1, (ws - 1) / 2}, m)
      m = torch.FloatTensor(m)
      cv.warp_affine(src, dst, m)
      dst:mul(contrast):add(brightness)
   end

   -- subset training dataset
   if opt.subset < 1 then
      function sample(xs, p)
         local perm = torch.randperm(xs:nElement()):long()
         return xs:index(1, perm[{{1, xs:size(1) * p}}])
      end

      local tr_subset
      if dataset == 'kitti' or dataset == 'kitti2015' then
         tr_subset = sample(tr, opt.subset)
      elseif dataset == 'mb' then
         tr_2014 = sample(torch.range(11, 23):long(), opt.subset)
         tr_2006 = sample(torch.range(24, 44):long(), opt.subset)
         tr_2005 = sample(torch.range(45, 50):long(), opt.subset)
         tr_2003 = sample(torch.range(51, 52):long(), opt.subset)
         tr_2001 = sample(torch.range(53, 60):long(), opt.subset)

         tr_subset = torch.cat(tr_2014, tr_2006)
         tr_subset = torch.cat(tr_subset, tr_2005)
         tr_subset = torch.cat(tr_subset, tr_2003)
         tr_subset = torch.cat(tr_subset, tr_2001)
      end

      local nnz_tr_output = torch.FloatTensor(nnz_tr:size()):zero()
      local t = adcensus.subset_dataset(tr_subset, nnz_tr, nnz_tr_output);
      nnz_tr = nnz_tr_output[{{1,t}}]
   end
   collectgarbage()

   if opt.a == 'train_all' then
      nnz = torch.cat(nnz_tr, nnz_te, 1)
   elseif opt.a == 'train_tr' or opt.a == 'time' then
      nnz = nnz_tr
   end

   if opt.a ~= 'time' then
      perm = torch.randperm(nnz:size(1))
   end

   local fm = torch.totable(torch.linspace(opt.fm, opt.fm, opt.l1):int())

   -- network for training
   if arch == 'slow' then
      net_tr = nn.Sequential()
      for i = 1,#fm do
         net_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
         net_tr:add(cudnn.ReLU(true))
      end
      net_tr:add(nn.Reshape(opt.bs, 2 * fm[#fm]))
      for i = 1,opt.l2 do
         net_tr:add(nn.Linear(i == 1 and 2 * fm[#fm] or opt.nh2, opt.nh2))
         net_tr:add(cudnn.ReLU(true))
      end
      net_tr:add(nn.Linear(opt.nh2, 1))
      net_tr:add(cudnn.Sigmoid(false))
      net_tr:cuda()
      criterion = nn.BCECriterion2():cuda()

      -- network for testing (make sure it's synched with net_tr)
      local pad = (opt.ks - 1) / 2
      net_te = nn.Sequential()
      for i = 1,#fm do
         net_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
         net_te:add(cudnn.ReLU(true))
      end
      net_te:cuda()

      net_te2 = nn.Sequential()
      for i = 1,opt.l2 do
         net_te2:add(nn.SpatialConvolution1_fw(i == 1 and 2 * fm[#fm] or opt.nh2, opt.nh2))
         net_te2:add(cudnn.ReLU(true))
      end
      net_te2:add(nn.SpatialConvolution1_fw(opt.nh2, 1))
      net_te2:add(cudnn.Sigmoid(true))
      net_te2:cuda()

      -- tie weights
      net_te_all = {}
      for i, v in ipairs(net_te.modules) do table.insert(net_te_all, v) end
      for i, v in ipairs(net_te2.modules) do table.insert(net_te_all, v) end

      local finput = torch.CudaTensor()
      local i_tr = 1
      local i_te = 1
      while i_tr <= net_tr:size() do
         local module_tr = net_tr:get(i_tr)
         local module_te = net_te_all[i_te]

         local skip = {['nn.Reshape']=1, ['nn.Dropout']=1}
         while skip[torch.typename(module_tr)] do
            i_tr = i_tr + 1
            module_tr = net_tr:get(i_tr)
         end

         if module_tr.weight then
            -- print(('tie weights of %s and %s'):format(torch.typename(module_te), torch.typename(module_tr)))
            assert(module_te.weight:nElement() == module_tr.weight:nElement())
            assert(module_te.bias:nElement() == module_tr.bias:nElement())
            module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
            module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
         end

         i_tr = i_tr + 1
         i_te = i_te + 1
      end
   elseif arch == 'fast' then
      net_tr = nn.Sequential()
      for i = 1,#fm do
         net_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
         if i < #fm then
            net_tr:add(cudnn.ReLU(true))
         end
      end
      net_tr:add(nn.Normalize2())
      net_tr:add(nn.StereoJoin1())
      net_tr:cuda()

      net_te = net_tr:clone('weight', 'bias')
      net_te.modules[#net_te.modules] = nn.StereoJoin(1):cuda()
      for i = 1,#net_te.modules do
         local m = net_te:get(i)
         if torch.typename(m) == 'cudnn.SpatialConvolution' then
            m.padW = 1
            m.padH = 1
         end
      end

      criterion = nn.Margin2(opt.m, opt.pow):cuda()
   end

   print_net(net_tr)

   params = {}
   grads = {}
   momentums = {}
   for i = 1,net_tr:size() do
      local m = net_tr:get(i)
      if m.weight then
         m.weight_v = torch.CudaTensor(m.weight:size()):zero()
         table.insert(params, m.weight)
         table.insert(grads, m.gradWeight)
         table.insert(momentums, m.weight_v)
      end
      if m.bias then
         m.bias_v = torch.CudaTensor(m.bias:size()):zero()
         table.insert(params, m.bias)
         table.insert(grads, m.gradBias)
         table.insert(momentums, m.bias_v)
      end
   end

   ws = get_window_size(net_tr)
   x_batch_tr = torch.CudaTensor(opt.bs * 2, n_input_plane, ws, ws)
   y_batch_tr = torch.CudaTensor(opt.bs)
   x_batch_tr_ = torch.FloatTensor(x_batch_tr:size())
   y_batch_tr_ = torch.FloatTensor(y_batch_tr:size())

   time = sys.clock()
   for epoch = 1,14 do
      if opt.a == 'time' then
         break
      end
      if epoch == 12 then
         opt.lr = opt.lr / 10
      end

      local err_tr = 0
      local err_tr_cnt = 0
      for t = 1,nnz:size(1) - opt.bs/2,opt.bs/2 do
         for i = 1,opt.bs/2 do
            d_pos = torch.uniform(-opt.true1, opt.true1)
            d_neg = torch.uniform(opt.false1, opt.false2)
            if torch.uniform() < 0.5 then
               d_neg = -d_neg
            end

            assert(opt.hscale <= 1 and opt.scale <= 1)
            local s = torch.uniform(opt.scale, 1)
            local scale = {s * torch.uniform(opt.hscale, 1), s}
            if opt.hflip == 1 and torch.uniform() < 0.5 then
               scale[1] = -scale[1]
            end
            if opt.vflip == 1 and torch.uniform() < 0.5 then
               scale[2] = -scale[2]
            end
            local hshear = torch.uniform(-opt.hshear, opt.hshear)
            local trans = {torch.uniform(-opt.trans, opt.trans), torch.uniform(-opt.trans, opt.trans)}
            local phi = torch.uniform(-opt.rotate * math.pi / 180, opt.rotate * math.pi / 180)
            local brightness = torch.uniform(-opt.brightness, opt.brightness)
            assert(opt.contrast >= 1 and opt.d_contrast >= 1)
            local contrast = torch.uniform(1 / opt.contrast, opt.contrast)

            local scale_ = {scale[1] * torch.uniform(opt.d_hscale, 1), scale[2]}
            local hshear_ = hshear + torch.uniform(-opt.d_hshear, opt.d_hshear)
            local trans_ = {trans[1], trans[2] + torch.uniform(-opt.d_vtrans, opt.d_vtrans)}
            local phi_ = phi + torch.uniform(-opt.d_rotate * math.pi / 180, opt.d_rotate * math.pi / 180)
            local brightness_ = brightness + torch.uniform(-opt.d_brightness, opt.d_brightness)
            local contrast_ = contrast * torch.uniform(1 / opt.d_contrast, opt.d_contrast)

            local ind = perm[t + i - 1]
            img = nnz[{ind, 1}]
            dim3 = nnz[{ind, 2}]
            dim4 = nnz[{ind, 3}]
            d = nnz[{ind, 4}]
            if dataset == 'kitti' or dataset == 'kitti2015' then
               x0 = X0[img]
               x1 = X1[img]
            elseif dataset == 'mb' then
               light = (torch.random() % (#X[img] - 1)) + 2
               exp = (torch.random() % X[img][light]:size(1)) + 1
               light_ = light
               exp_ = exp
               if torch.uniform() < opt.d_exp then
                  exp_ = (torch.random() % X[img][light]:size(1)) + 1
               end
               if torch.uniform() < opt.d_light then
                  light_ = math.max(2, light - 1)
               end
               x0 = X[img][light][{exp,1}]
               x1 = X[img][light_][{exp_,2}]
            end

            make_patch(x0, x_batch_tr_[i * 4 - 3], dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
            make_patch(x1, x_batch_tr_[i * 4 - 2], dim3, dim4 - d + d_pos, scale_, phi_, trans_, hshear_, brightness_, contrast_)
            make_patch(x0, x_batch_tr_[i * 4 - 1], dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
            make_patch(x1, x_batch_tr_[i * 4 - 0], dim3, dim4 - d + d_neg, scale_, phi_, trans_, hshear_, brightness_, contrast_)

            y_batch_tr_[i * 2 - 1] = 0
            y_batch_tr_[i * 2] = 1
         end

         x_batch_tr:copy(x_batch_tr_)
         y_batch_tr:copy(y_batch_tr_)

         for i = 1,#params do
            grads[i]:zero()
         end

         net_tr:forward(x_batch_tr)
         local err = criterion:forward(net_tr.output, y_batch_tr)
         if err >= 0 and err < 100 then
            err_tr = err_tr + err
            err_tr_cnt = err_tr_cnt + 1
         else
            print(('WARNING! err=%f'):format(err))
         end

         criterion:backward(net_tr.output, y_batch_tr)
         net_tr:backward(x_batch_tr, criterion.gradInput)

         for i = 1,#params do
            momentums[i]:mul(opt.mom):add(-opt.lr, grads[i])
            params[i]:add(momentums[i])
         end
      end

      if opt.debug then
         save_net(epoch)
      end
      print(epoch, err_tr / err_tr_cnt, opt.lr, sys.clock() - time)
      collectgarbage()
   end
   opt.net_fname = save_net(0)
   if opt.a == 'train_tr' then
      opt.a = 'test_te'
   elseif opt.a == 'train_all' then
      opt.a = 'submit'
   end
   collectgarbage()
end

if not opt.use_cache then
   if arch == 'slow' then
      net = torch.load(opt.net_fname, 'ascii')
      net_te = net[1]
      net_te2 = net[2]
   elseif arch == 'fast' then
      net_te = torch.load(opt.net_fname, 'ascii')[1]
      net_te.modules[#net_te.modules] = nil
      net_te2 = nn.StereoJoin(1):cuda()
   end
end

x_batch_te1 = torch.CudaTensor()
x_batch_te2 = torch.CudaTensor()

function forward_free(net, input)
   local currentOutput = input
   for i=1,#net.modules do
      net.modules[i].oDesc = nil
      local nextOutput = net.modules[i]:updateOutput(currentOutput)
      if currentOutput:storage() ~= nextOutput:storage() then
         currentOutput:storage():resize(1)
         currentOutput:resize(0)
      end
      currentOutput = nextOutput
   end
   net.output = currentOutput
   return currentOutput
end

function fix_border(net, vol, direction)
   local n = (get_window_size(net) - 1) / 2
   for i=1,n do
      vol[{{},{},{},direction * i}]:copy(vol[{{},{},{},direction * (n + 1)}])
   end
end

function stereo_predict(x_batch, id)
   local vols, vol

   if arch == 'ad' then
      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
      adcensus.ad(x_batch[{{1}}], x_batch[{{2}}], vols[{{1}}], -1)
      adcensus.ad(x_batch[{{2}}], x_batch[{{1}}], vols[{{2}}], 1)
   end

   if arch == 'census' then
      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
      adcensus.census(x_batch[{{1}}], x_batch[{{2}}], vols[{{1}}], -1)
      adcensus.census(x_batch[{{2}}], x_batch[{{1}}], vols[{{2}}], 1)
   end

   if arch == 'fast' then
      forward_free(net_te, x_batch:clone())
      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
      adcensus.StereoJoin(net_te.output[{{1}}], net_te.output[{{2}}], vols[{{1}}], vols[{{2}}])
      fix_border(net_te, vols[{{1}}], -1)
      fix_border(net_te, vols[{{2}}], 1)
      clean_net(net_te)
   end

   disp = {}
   local mb_directions = opt.a == 'predict' and {1, -1} or {-1}
   for _, direction in ipairs(dataset == 'mb' and mb_directions or {1, -1}) do
      sm_active = true

      if arch == 'slow' then
         if opt.use_cache then
            vol = torch.load(('cache/%s_%d.t7'):format(id, direction))
         else
            local output = forward_free(net_te, x_batch:clone())
            clean_net(net_te)
            collectgarbage()

            vol = torch.CudaTensor(1, disp_max, output:size(3), output:size(4)):fill(0 / 0)
            collectgarbage()
            for d = 1,disp_max do
               local l = output[{{1},{},{},{d,-1}}]
               local r = output[{{2},{},{},{1,-d}}]
               x_batch_te2:resize(2, r:size(2), r:size(3), r:size(4))
               x_batch_te2[1]:copy(l)
               x_batch_te2[2]:copy(r)
               x_batch_te2:resize(1, 2 * r:size(2), r:size(3), r:size(4))
               forward_free(net_te2, x_batch_te2)
               vol[{1,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(net_te2.output[{1,1}])
            end
            clean_net(net_te2)
            fix_border(net_te, vol, direction)
            if opt.make_cache then
               torch.save(('cache/%s_%d.t7'):format(id, direction), vol)
            end
         end
         collectgarbage()
      elseif arch == 'fast' or arch == 'ad' or arch == 'census' then
         vol = vols[{{direction == -1 and 1 or 2}}]
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cnn')

      -- cross computation
      local x0c, x1c
      if sm_active and opt.sm_skip ~= 'cbca' then
         x0c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
         x1c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
         adcensus.cross(x_batch[1], x0c, opt.L1, opt.tau1)
         adcensus.cross(x_batch[2], x1c, opt.L1, opt.tau1)
         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
         for i = 1,opt.cbca_i1 do
            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
            vol:copy(tmp_cbca)
         end
         tmp_cbca = nil
         collectgarbage()
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cbca1')

      if sm_active and opt.sm_skip ~= 'sgm' then
         vol = vol:transpose(2, 3):transpose(3, 4):clone()
         collectgarbage()
         do
            local out = torch.CudaTensor(1, vol:size(2), vol:size(3), vol:size(4))
            local tmp = torch.CudaTensor(vol:size(3), vol:size(4))
            for _ = 1,opt.sgm_i do
               out:zero()
               adcensus.sgm2(x_batch[1], x_batch[2], vol, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
                  opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction)
               vol:copy(out):div(4)
            end
            vol:resize(1, disp_max, x_batch:size(3), x_batch:size(4))
            vol:copy(out:transpose(3, 4):transpose(2, 3)):div(4)

--            local out = torch.CudaTensor(4, vol:size(2), vol:size(3), vol:size(4))
--            out:zero()
--            adcensus.sgm3(x_batch[1], x_batch[2], vol, out, opt.pi1, opt.pi2, opt.tau_so,
--               opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction)
--            vol:mean(out, 1)
--            vol = vol:transpose(3, 4):transpose(2, 3):clone()
         end
         collectgarbage()
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'sgm')

      if sm_active and opt.sm_skip ~= 'cbca' then
         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
         for i = 1,opt.cbca_i2 do
            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
            vol:copy(tmp_cbca)
         end
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cbca2')

      if opt.a == 'predict' then
         local fname = direction == -1 and 'left' or 'right'
         print(('Writing %s.bin, %d x %d x %d x %d'):format(fname, vol:size(1), vol:size(2), vol:size(3), vol:size(4)))
         torch.DiskFile(('%s.bin'):format(fname), 'w'):binary():writeFloat(vol:float():storage())
         collectgarbage()
      end

      _, d = torch.min(vol, 2)
      disp[direction == 1 and 1 or 2] = d:cuda():add(-1)
   end
   collectgarbage()

   if dataset == 'kitti' or dataset == 'kitti2015' then
      local outlier = torch.CudaTensor():resizeAs(disp[2]):zero()
      adcensus.outlier_detection(disp[2], disp[1], outlier, disp_max)
      if sm_active and opt.sm_skip ~= 'occlusion' then
         disp[2] = adcensus.interpolate_occlusion(disp[2], outlier)
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'occlusion')

      if sm_active and opt.sm_skip ~= 'occlusion' then
         disp[2] = adcensus.interpolate_mismatch(disp[2], outlier)
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'mismatch')
   end
   if sm_active and opt.sm_skip ~= 'subpixel_enchancement' then
      disp[2] = adcensus.subpixel_enchancement(disp[2], vol, disp_max)
   end
   sm_active = sm_active and (opt.sm_terminate ~= 'subpixel_enchancement')

   if sm_active and opt.sm_skip ~= 'median' then
      disp[2] = adcensus.median2d(disp[2], 5)
   end
   sm_active = sm_active and (opt.sm_terminate ~= 'median')

   if sm_active and opt.sm_skip ~= 'bilateral' then
      disp[2] = adcensus.mean2d(disp[2], gaussian(opt.blur_sigma):cuda(), opt.blur_t)
   end

   return disp[2]
end

if opt.a == 'predict' then
   x0 = image.load(opt.left, nil, 'byte'):float()
   x1 = image.load(opt.right, nil, 'byte'):float()

   if x0:size(1) == 3 then
      assert(x1:size(1) == 3)
      x0 = image.rgb2y(x0)
      x1 = image.rgb2y(x1)
   end
   disp_max = opt.disp_max

   x0:add(-x0:mean()):div(x0:std())
   x1:add(-x1:mean()):div(x1:std())

   x_batch = torch.CudaTensor(2, 1, x0:size(2), x0:size(3))
   x_batch[1]:copy(x0)
   x_batch[2]:copy(x1)
   disp = stereo_predict(x_batch, 0)
   print(('Writing disp.bin, %d x %d x %d x %d'):format(disp:size(1), disp:size(2), disp:size(3), disp:size(4)))
   torch.DiskFile('disp.bin', 'w'):binary():writeFloat(disp:float():storage())
   os.exit()
end

if opt.a == 'submit' then
   os.execute('rm -rf out/*')
   if dataset == 'kitti2015' then
      os.execute('mkdir out/disp_0')
   end
   if dataset == 'kitti' or dataset == 'kitti2015' then
      examples = torch.totable(torch.range(X0:size(1) - n_te + 1, X0:size(1)))
   elseif dataset == 'mb' then
      examples = {}
      -- for i = #X - 14, #X do
      for i = #X - 29, #X do
         table.insert(examples, {i, 2})
      end
   end
elseif opt.a == 'test_te' then
   if dataset == 'kitti' or dataset == 'kitti2015' then
      examples = torch.totable(te)
   elseif dataset == 'mb' then
      examples = {}
      for i = 1,te:nElement() do
         table.insert(examples, {te[i], 2})
      end
      table.insert(examples, {5, 3})
      table.insert(examples, {5, 4})
   end
elseif opt.a == 'test_all' then
   if dataset == 'kitti' or dataset == 'kitti2015' then
      examples = torch.totable(torch.cat(tr, te))
   elseif dataset == 'mb' then
      assert(false, 'test_all not supported on Middlebury.')
   end
end

if opt.a == 'time' then
   if opt.tiny then
      x_batch = torch.CudaTensor(2, 1, 240, 320)
      disp_max = 32
   elseif dataset == 'kitti' then
      x_batch = torch.CudaTensor(2, 1, 350, 1242)
      disp_max = 228
   elseif dataset == 'mb' then
      x_batch = torch.CudaTensor(2, 1, 1000, 1500)
      disp_max = 200
   end

   N = arch == 'fast' and 30 or 3
   runtime_min = 1 / 0
   for _ = 1,N do
      cutorch.synchronize()
      sys.tic()

      stereo_predict(x_batch, id)

      cutorch.synchronize()
      runtime = sys.toc()
      if runtime < runtime_min then
         runtime_min = runtime
      end
      collectgarbage()
   end
   print(runtime_min)

   os.exit()
end

err_sum = 0
x_batch = torch.CudaTensor(2, 1, height, width)
pred_good = torch.CudaTensor()
pred_bad = torch.CudaTensor()
for _, i in ipairs(examples) do
   if dataset == 'kitti' or dataset == 'kitti2015' then
      img_height = metadata[{i,1}]
      img_width = metadata[{i,2}]
      id = metadata[{i,3}]
      x0 = X0[{{i},{},{},{1,img_width}}]
      x1 = X1[{{i},{},{},{1,img_width}}]
   elseif dataset == 'mb' then
      i, right = table.unpack(i)
      id = ('%d_%d'):format(i, right)
      disp_max = metadata[{i,3}]
      x0 = X[i][1][{{1}}]
      x1 = X[i][1][{{right}}]
   end

   x_batch:resize(2, 1, x0:size(3), x0:size(4))
   x_batch[1]:copy(x0)
   x_batch[2]:copy(x1)

   collectgarbage()
   cutorch.synchronize()
   sys.tic()
   pred = stereo_predict(x_batch, id)
   cutorch.synchronize()
   runtime = sys.toc()

   if opt.a == 'submit' then
      if dataset == 'kitti' or dataset == 'kitti2015' then
         pred_img = torch.FloatTensor(img_height, img_width):zero()
         pred_img:narrow(1, img_height - height + 1, height):copy(pred[{1,1}])
        
         if dataset == 'kitti' then
            path = 'out'
         elseif dataset == 'kitti2015' then
            path = 'out/disp_0'
         end
         adcensus.writePNG16(pred_img, img_height, img_width, ("%s/%06d_10.png"):format(path, id))
      elseif dataset == 'mb' then
         -- savePNG(('tmp/fos_%d.png'):format(i), pred)
         base = 'out/' .. fname_submit[i - (#X - #fname_submit)]
         os.execute('mkdir -p ' .. base)
         local method_name = 'MC-CNN-' .. (arch == 'fast' and 'fst' or 'acrt' )
         adcensus.writePFM(image.vflip(pred[{1,1}]:float()), base .. '/disp0' .. method_name .. '.pfm')
         local f = io.open(base .. '/time' .. method_name .. '.txt', 'w')
         f:write(tostring(runtime))
         f:close()
      end
   else
      assert(not isnan(pred:sum()))
      if dataset == 'kitti' or dataset == 'kitti2015' then
         actual = dispnoc[{i,{},{},{1,img_width}}]:cuda()
      elseif dataset == 'mb' then
         actual = dispnoc[i]:cuda()
      end
      pred_good:resizeAs(actual)
      pred_bad:resizeAs(actual)
      mask = torch.CudaTensor():resizeAs(actual):ne(actual, 0)
      actual:add(-1, pred):abs()
      pred_bad:gt(actual, err_at):cmul(mask)
      pred_good:le(actual, err_at):cmul(mask)
      local err = pred_bad:sum() / mask:sum()
      err_sum = err_sum + err
      print(runtime, err)

      if opt.debug then
         local img_pred = torch.Tensor(1, 3, pred:size(3), pred:size(4))
         adcensus.grey2jet(pred:double():add(1)[{1,1}]:div(disp_max):double(), img_pred)
         if x0:size(2) == 1 then
            x0 = torch.repeatTensor(x0:cuda(), 1, 3, 1, 1)
         end
         img_err = x0:mul(50):add(150):div(255)
         img_err[{1,1}]:add( 0.5, pred_bad)
         img_err[{1,2}]:add(-0.5, pred_bad)
         img_err[{1,3}]:add(-0.5, pred_bad)
         img_err[{1,1}]:add(-0.5, pred_good)
         img_err[{1,2}]:add( 0.5, pred_good)
         img_err[{1,3}]:add(-0.5, pred_good)

         if dataset == 'kitti' or dataset == 'kitti2015' then
            gt = dispnoc[{{i},{},{},{1,img_width}}]:cuda()
         elseif dataset == 'mb' then
            gt = dispnoc[i]:cuda():resize(1, 1, pred:size(3), pred:size(4))
         end
         local img_gt = torch.Tensor(1, 3, pred:size(3), pred:size(4)):zero()
         adcensus.grey2jet(gt:double():add(1)[{1,1}]:div(disp_max):double(), img_gt)
         img_gt[{1,3}]:cmul(mask:double())

         image.save(('tmp/%s_%s_gt.png'):format(dataset, id), img_gt[1])
         image.save(('tmp/%s_%s_%s_pred.png'):format(dataset, arch, id), img_pred[1])
         image.save(('tmp/%s_%s_%s_err.png'):format(dataset, arch, id), img_err[1])

--         adcensus.grey2jet(pred:double():add(1)[{1,1}]:div(disp_max):double(), img_pred)
--         if x0:size(2) == 1 then
--            x0 = torch.repeatTensor(x0:cuda(), 1, 3, 1, 1)
--         end
--         img_err = x0:mul(50):add(150):div(255)
--         img_err[{1,1}]:add( 0.3, pred_bad)
--         img_err[{1,2}]:add(-0.3, pred_bad)
--         img_err[{1,3}]:add(-0.3, pred_bad)
--         img_err[{1,1}]:add(-0.3, pred_good)
--         img_err[{1,2}]:add( 0.3, pred_good)
--         img_err[{1,3}]:add(-0.3, pred_good)
--
--         img = torch.Tensor(3, pred:size(3) * 2, pred:size(4))
--         img:narrow(2, 0 * pred:size(3) + 1, pred:size(3)):copy(img_pred)
--         img:narrow(2, 1 * pred:size(3) + 1, pred:size(3)):copy(img_err)
--
--         image.savePNG(('tmp/err_%d_%.5f_%s.png'):format(opt.gpu, err, id), img)
      end
   end
end

if opt.a == 'submit' then
   -- zip
   os.execute('cd out; zip -r submission.zip . -x .empty')
else
   print(err_sum / #examples)
end
