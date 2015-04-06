-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Please find submission instructions on the handout
------------------------------------------------------------------------

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   self.exps = torch.exp(self.beta * input) 
   self.sumexp = torch.sum(self.exps)
   self.output = torch.log(self.sumexp/input:size(1))/self.beta
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   vself.buffer = self.buffer or input.new()
   self.buffer:resizeAs(input):copy(input)
   self.buffer:mul(self.beta):exp()
   self.gradInput:resizeAs(input):copy(gradOutput):cmul(self.exps/(self.beta * self.beta * self.sumexp * input:size(1)))
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
   self.sumexp:resize()
   self.sumexp:storage:resize(0)
   self.exps:resize()
   self.exps:storage:resize(0)
end
