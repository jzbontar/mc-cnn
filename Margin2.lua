local Margin2, parent = torch.class('nn.Margin2', 'nn.Criterion')

function Margin2:__init(margin, pow)
   parent.__init(self)
   self.tmp = torch.CudaTensor()
   self.margin = margin
   self.pow = pow
end

function Margin2:updateOutput(input, target)
   assert(input:size(2) == 1 and input:size(3) == 1 and input:size(4) == 1)
   self.tmp:resize(input:size(1) / 2)
   self.gradInput:resizeAs(input)
   adcensus.Margin2(input, self.tmp, self.gradInput, self.margin, self.pow)
   self.output = self.tmp:mean()
   self.gradInput:div(self.tmp:size(1))
   return self.output
end

function Margin2:updateGradInput(input, target)
   return self.gradInput
end
