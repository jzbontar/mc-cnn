#! /usr/bin/env luajit

require 'Test'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'libadcensus'

include('Margin2.lua')
include('StereoJoin1.lua')
include('StereoJoin.lua')
include('Normalize.lua')

function test_StereoJoin1()
   print('test_StereoJoin1')
   input_ = torch.Tensor(6, 4, 5, 6):normal()

   -- forward
   output_ = torch.Tensor(input_:size(1) / 2, 1, input_:size(3), input_:size(4))
   for dim1 = 1, output_:size(1) do
      for dim3 = 1, output_:size(3) do
         for dim4 = 1, output_:size(4) do
            local sum = 0
            for dim2_ = 1, input_:size(2) do
               left = input_[{dim1 * 2 - 1,dim2_,dim3,dim4}]
               right = input_[{dim1 * 2,dim2_,dim3,dim4}]
               sum = sum + left * right
            end
            output_[{dim1,1,dim3,dim4}] = sum
         end
      end
   end

   module = nn.Sequential()
   module:add(nn.StereoJoin1(op))
   module:cuda()

   module:forward(input_:cuda())
   print(output_:add(-1, module.output:double()):abs():max())

   -- backward
   print(testJacobian(module, input_:cuda()))
end

function test_StereoJoin()
   print('test_StereoJoin')

   input_ = torch.Tensor(2, 32, 10, 20):normal()
   output_ = torch.Tensor(1, 16, 10, 20):zero()

   for dim2 = 1, output_:size(2) do
      for dim3 = 1, output_:size(3) do
         for dim4 = 1, output_:size(4) do
            if dim4 - dim2 + 1 <= 0 then
               output_[{1,dim2,dim3,dim4}] = 0 / 0
            else
               sum = 0.0
               for dim2_ = 1, input_:size(2) do
                  left = input_[{1,dim2_,dim3,dim4}]
                  right = input_[{2,dim2_,dim3,dim4 - dim2 + 1}]
                  sum = sum + left * right
               end
               output_[{1,dim2,dim3,dim4}] = sum
            end
         end
      end
   end

   module = nn.StereoJoin(output_:size(2)):cuda()
   module:forward(input_:cuda())

   print(output_:add(-1, module.output:double()):abs():max())
end

test_StereoJoin()

function test_Normalize()
   print('test_Normalize')
   input_ = torch.Tensor(2, 3, 4, 5):normal()
   output_ = torch.Tensor(2, 3, 4, 5):normal()
   norm_ = torch.Tensor(2, 1, 4, 5):zero()

   -- forward
   for dim1 = 1, output_:size(1) do
      for dim3 = 1, output_:size(3) do
         for dim4 = 1, output_:size(4) do
            sum = 0.0
            for dim2 = 1, output_:size(2) do
               x = input_[{dim1,dim2,dim3,dim4}]
               sum = sum + x * x
            end
            norm_[{dim1,1,dim3,dim4}] = sum
            for dim2 = 1, output_:size(2) do
               output_[{dim1,dim2,dim3,dim4}] = input_[{dim1,dim2,dim3,dim4}] / math.sqrt(sum)
            end
         end
      end
   end

   module = nn.Normalize():cuda()
   module:forward(input_:cuda())

   print(norm_:add(-1, module.norm:double()):abs():max())
   print(output_:add(-1, module.output:double()):abs():max())

   -- backward
   print(testJacobian(module, input_:cuda()))
end


function test_Margin2()
   print('test_Margin2')

   margin = 0.1
   pow = 2
   input_ = torch.Tensor(64, 1, 1, 1):uniform()
   tmp_ = torch.Tensor(32, 1, 1, 1):zero()

   -- forward
   for dim1 = 1, tmp_:size(1) do
      d = math.max(0, input_[{dim1 * 2,1,1,1}] - input_[{dim1 * 2 - 1,1,1,1}] + margin)
      if pow == 2 then
         d = d * d * 0.5
      end
      tmp_[dim1] = d
   end

   module = nn.Margin2(margin, pow):cuda()
   module:forward(input_:cuda(), nil)

   print(tmp_:add(-1, module.tmp:double()):abs():max())

   -- backward
   print(testCriterion(module, input_:cuda(), nil))
end
