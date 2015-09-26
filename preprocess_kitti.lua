#!/usr/bin/env luajit

require 'image'
require 'nn'
require 'cutorch'
require 'libadcensus'

for _, dataset in ipairs({2012, 2015}) do
   print(('dataset %d'):format(dataset))

   torch.manualSeed(42)
   if dataset == 2012 then
      n_tr = 194
      n_te = 195
      path = 'data.kitti'
      image_0 = 'image_0'
      image_1 = 'image_1'
      disp_noc = 'disp_noc'
      nchannel = 1
   elseif dataset == 2015 then
      n_tr = 200
      n_te = 200
      path = 'data.kitti2015'
      image_0 = 'image_2'
      image_1 = 'image_3'
      nchannel = 3
      disp_noc = 'disp_noc_0'
   end

   height = 350
   width = 1242

   x0 = torch.FloatTensor(n_tr + n_te, 1, height, width):zero()
   x1 = torch.FloatTensor(n_tr + n_te, 1, height, width):zero()
   dispnoc = torch.FloatTensor(n_tr, 1, height, width):zero()
   metadata = torch.IntTensor(n_tr + n_te, 3):zero()

   examples = {}
   for i = 1,n_tr do
      examples[#examples + 1] = {dir='training', cnt=i}
   end

   for i = 1,n_te do
      examples[#examples + 1] = {dir='testing', cnt=i}
   end

   for i, arg in ipairs(examples) do
      img_path = '%s/unzip/%s/%s/%06d_10.png'
      img_0 = image.loadPNG(img_path:format(path, arg['dir'], image_0, arg['cnt'] - 1), nchannel, 'byte'):float()
      img_1 = image.loadPNG(img_path:format(path, arg['dir'], image_1, arg['cnt'] - 1), nchannel, 'byte'):float()

      if dataset == 2015 then
         img_0 = image.rgb2y(img_0)
         img_1 = image.rgb2y(img_1)
      end

      -- crop
      img_height = img_0:size(2)
      img_width = img_0:size(3)
      img_0 = img_0:narrow(2, img_height - height + 1, height)
      img_1 = img_1:narrow(2, img_height - height + 1, height)

      -- preprocess
      print(i)

      img_0:add(-img_0:mean()):div(img_0:std())
      img_1:add(-img_1:mean()):div(img_1:std())

      x0[{i,{},{},{1,img_width}}]:copy(img_0)
      x1[{i,{},{},{1,img_width}}]:copy(img_1)

      if arg['dir'] == 'training' then
         img_disp = torch.FloatTensor(1, img_height, img_width)
         adcensus.readPNG16(img_disp, ('%s/unzip/training/%s/%06d_10.png'):format(path, disp_noc, arg['cnt'] - 1))
         dispnoc[{i, 1}]:narrow(2, 1, img_width):copy(img_disp:narrow(2, img_height - height + 1, height))
      end

      metadata[{i, 1}] = img_height
      metadata[{i, 2}] = img_width
      metadata[{i, 3}] = arg['cnt'] - 1

      collectgarbage()
   end

   -- split train and test
   perm = torch.randperm(n_tr):long()
   te = perm[{{1,40}}]:clone()
   tr = perm[{{41,n_tr}}]:clone()

   -- prepare tr dataset
   nnz_tr = torch.FloatTensor(23e6, 4)
   nnz_te = torch.FloatTensor(23e6, 4)
   nnz_tr_t = 0
   nnz_te_t = 0
   for i = 1,n_tr do
      local disp = dispnoc[{{i}}]:cuda()
      adcensus.remove_nonvisible(disp)
      adcensus.remove_occluded(disp)
      adcensus.remove_white(x0[{{i}}]:cuda(), disp)
      disp = disp:float()

      is_te = false
      for j = 1,te:nElement() do
         if i == te[j] then
            is_te = true
         end
      end

      if is_te then
         nnz_te_t = adcensus.make_dataset2(disp, nnz_te, i, nnz_te_t)
      else
         nnz_tr_t = adcensus.make_dataset2(disp, nnz_tr, i, nnz_tr_t)
      end
   end
   nnz_tr = torch.FloatTensor(nnz_tr_t, 4):copy(nnz_tr[{{1,nnz_tr_t}}])
   nnz_te = torch.FloatTensor(nnz_te_t, 4):copy(nnz_te[{{1,nnz_te_t}}])

   function tofile(fname, x)
      tfile = torch.DiskFile(fname .. '.type', 'w')
      if x:type() == 'torch.FloatTensor' then
         tfile:writeString('float32')
         torch.DiskFile(fname, 'w'):binary():writeFloat(x:storage())
      elseif x:type() == 'torch.LongTensor' then
         tfile:writeString('int64')
         torch.DiskFile(fname, 'w'):binary():writeLong(x:storage())
      elseif x:type() == 'torch.IntTensor' then
         tfile:writeString('int32')
         torch.DiskFile(fname, 'w'):binary():writeInt(x:storage())
      end
      dimfile = torch.DiskFile(fname .. '.dim', 'w')
      for i = 1,x:dim() do
         dimfile:writeString(('%d\n'):format(x:size(i)))
      end
   end

   os.execute(('rm -f %s/*.{bin,dim,type}'):format(path))
   tofile(('%s/x0.bin'):format(path), x0)
   tofile(('%s/x1.bin'):format(path), x1)
   tofile(('%s/dispnoc.bin'):format(path), dispnoc)
   tofile(('%s/metadata.bin'):format(path), metadata)
   tofile(('%s/tr.bin'):format(path), tr)
   tofile(('%s/te.bin'):format(path), te)
   tofile(('%s/nnz_tr.bin'):format(path), nnz_tr)
   tofile(('%s/nnz_te.bin'):format(path), nnz_te)
end
