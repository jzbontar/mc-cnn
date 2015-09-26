local StereoJoin, parent = torch.class('nn.StereoJoin', 'nn.Module')

function StereoJoin:__init(disp_max)
   parent.__init(self)
   self.direction = -1
   self.disp_max = disp_max
   self.output_L = torch.CudaTensor()
end

function StereoJoin:updateOutput(input)
   assert(input:size(1) == 2)
   local input_L = input[{{1}}]
   local input_R = input[{{2}}]
   self.output_L:resize(1, self.disp_max, input_L:size(3), input_L:size(4))
   adcensus.StereoJoin_forward2(input_L, input_R, self.output_L)
   return self.output_L
end
