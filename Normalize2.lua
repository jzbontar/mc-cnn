local Normalize2, parent = torch.class('nn.Normalize2', 'nn.Module')

function Normalize2:__init()
   parent.__init(self)
   self.norm = torch.CudaTensor()
end

function Normalize2:updateOutput(input)
   self.norm:resize(input:size(1), 1, input:size(3), input:size(4))
   self.output:resizeAs(input)
   adcensus.Normalize_forward(input, self.norm, self.output)
   return self.output
end

function Normalize2:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   adcensus.Normalize_backward_input(gradOutput, input, self.norm, self.gradInput)
   return self.gradInput
end
