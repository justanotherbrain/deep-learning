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
   self.sums = torch.Tensor()
   self.buffer = torch.Tensor()
   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   --first dimension = # time steps
   --second dimension = feature
   --Handle 2-3 dimensional input
   local inp
   local nOutputFrame
   if input:size():size() == 2 then
     inp = input:reshape(1,input:size(1),input:size(2))
     nOutputFrame = (input:size(1) - self.kW) /self.dW + 1
   else
     inp = input
     nOutputFrame = (input:size(2) - self.kW) /self.dW + 1
   end
   self.nOutputFrame = math.floor(nOutputFrame)
   self.output = torch.zeros(inp:size(1), nOutputFrame,inp:size(3))
   self.sums = torch.zeros(self.output:size())
   for i=1,inp:size(1) do
     for j = 0, nOutputFrame-1 do
       start = self.dW * j + 1
       stop = start + self.kW - 1
       local intemp = inp[{i,{start,stop},{}}]
       local exps = torch.exp(intemp * self.beta)
       local sumexp = exps:sum(1)
       local out = torch.log(sumexp/self.kW)/self.beta
       self.sums[{i,{j+1},{}}] = sumexp
       self.output[{i,{j+1},{}}] = out
     end
   end
   --Fix output view for 2 dimensional output
   if input:size():size() == 2 then
     self.output = self.output:reshape(nOutputFrame, input:size(2))
   end
   --SUMS IS ACTUALLY 1/SUM
   self.sums = self.sums:pow(-1)
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   local inp
   local gradOut
   if input:size():size() == 2 then
     inp = input:reshape(1,input:size(1), input:size(2))
     gradOut = gradOutput:reshape(1,gradOutput:size(1),gradOutput:size(2))
   else
     inp = input
     gradOut = gradOutput
   end

   local outRowsToUpdate = {}
   for inputRow = 1,inp:size(2) do
     table.insert(outRowsToUpdate,{})
     for outRow = 1,self.nOutputFrame do
        local outBot = self.dW * (outRow - 1) + 1
        local outTop = self.dW * (outRow - 1) + self.kW
        if inputRow >= outBot and inputRow <= outTop then
          table.insert(outRowsToUpdate[inputRow], outRow)
        end
     end
   end
   self.gradInput = torch.zeros(inp:size())
   for sample=1,inp:size(1) do
     for inputRow=1, inp:size(2) do
       for inputCol=1, inp:size(3) do
         local tempSum = 0
         for outRowInd =1,#outRowsToUpdate[inputRow] do
           local outRow = outRowsToUpdate[inputRow][outRowInd]
           tempSum = tempSum + self.sums[sample][outRow][inputCol] * torch.exp(inp[sample][inputRow][inputCol] * self.beta) * gradOut[sample][outRow][inputCol]
         end
         self.gradInput[sample][inputRow][inputCol] = tempSum 
       end
     end
   end
   if input:size():size() == 2 then
     self.gradInput = self.gradInput:reshape(input:size())
   end
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
   self.sums:resize()
   self.sums:storage():resize(0)
end
--first dimension = # time steps
--second dimension = feature
