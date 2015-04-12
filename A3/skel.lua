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

require 'nn'

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta
   self.model = nn.Sequential()
   
   self.model:add(nn.MulConstant(self.beta))
   
   self.model:add(nn.Exp())
   self.model:add(nn.Transpose())
   self.convLayer = nn.TemporalSubSampling(1, self.kW, self.dW)
   self.convLayer.reset = nil
   self.model:add(self.convLayer)
   self.model:add(nn.Transpose())

   self.model:add(self.MulConstant(1/self.kW))
   
   self.model:add(nn.Log())
   
   self.model:add(nn.MulConstant(1/self.beta))
end

function TemporalLogExpPooling:updateOutput(input)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   --first dimension = # time steps
   --second dimension = feature
   --Handle 2-3 dimensional input
   local inp
   if input:size():size() == 3 then
     inp = input
   else
     inp = input:reshape(1,input:size(1),input:size(2))
   end
   local ifs = inp:size(3)
   self.convLayer.inputFrameSize = ifs 
   self.convLayer.weight = torch.ones(ifs)
   self.convLayer.bias = torch.zeros(ifs)
   self.convLayer.gradWeight = self.convLayer.weight
   self.convLayer.gradBias = self.convLayer.bias
   self.output = torch.Tensor(inp:size(1),math.floor((inp:size(2)-self.kW))/self.dW + 1, inp:size(3))
   for i = 1,inp:size(1) do
     self.output[{i}] = self.model:forward(inp[i])
   end
   if input:size():size() == 2 then
     self.output:reshape(self.output:size(2), self.output:size(3))
   end
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- your code here
   ----------------------------------------------
   local outp
   local inp
   if gradOutput:size():size() == 3 then
     outp = gradOutput
     inp = input
   else
     outp = gradOutput:reshape(1,gradOutput:size(1),gradOutput:size(2))
     inp = input:reshape(1,input:size(1),gradOutput:size(2))
   end
   self.gradInput = torch.Tensor(inp:size()) 
   for i=1,outp:size(1) do
     self.gradInput[{i}] = self.model:backward(inp[i], outp[i])
   end
   if input:size():size() == 2 then
     self.gradInput:reshape(self.gradInput:size(2), self.gradInput:size(3))
   end
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
end
--first dimension = # time steps
--second dimension = feature
