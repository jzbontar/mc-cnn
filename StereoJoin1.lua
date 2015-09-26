local StereoJoin1, parent = torch.class('nn.StereoJoin1', 'nn.Module')

function StereoJoin1:__init()
   parent.__init(self)
   self.gradInput = torch.CudaTensor()
   self.tmp = torch.CudaTensor()
end

function slice_input(input)
   local sizes = torch.LongStorage{input:size(1) / 2, input:size(2), input:size(3), input:size(4)}
   local strides = torch.LongStorage{input:stride(1) * 2, input:stride(2), input:stride(3), input:stride(4)}
   local input_L = torch.CudaTensor(input:storage(), 1, sizes, strides)
   local input_R = torch.CudaTensor(input:storage(), input:stride(1) + 1, sizes, strides)
   return input_L, input_R
end

function StereoJoin1:updateOutput(input)
   local input_L, input_R = slice_input(input)
   self.tmp:resizeAs(input_L)
   self.tmp:cmul(input_L, input_R)
   self.output:sum(self.tmp, 2)
   return self.output
end

function StereoJoin1:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   local input_L, input_R = slice_input(input)
   local gradInput_L, gradInput_R = slice_input(self.gradInput)
   gradInput_L:cmul(input_R, gradOutput:expandAs(input_R))
   gradInput_R:cmul(input_L, gradOutput:expandAs(input_L))
   return self.gradInput
end
