----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      testSampleWrong[t]= testSampleWrong[t] + ((pred ~= target) and 1 or 0)--DG addition
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- DG addition. Print updated list of mislabeled samples every few iterations
   testWrite = {}
   trainWrite = {}
   for key, value in pairs(testSampleWrong) do
      table.insert(testWrite, {key,value})
   end
   for key, value in pairs(trainSampleWrong) do
      table.insert(trainWrite, {key,value})
   end
   table.insert(testWrite, {'Epoch',#testWrite+1})    
   table.insert(trainWrite, {'Epoch',#trainWrite+1})   
   table.sort(testWrite, sampleComparer)
   table.sort(trainWrite, sampleComparer)
   testWrite[1][2] = epoch - 1
   trainWrite[1][2] = epoch - 1
   csvigo.save{path=paths.concat(opt.save, 'test_wrongSamples.log'), data=testWrite}
   csvigo.save{path=paths.concat(opt.save, 'train_wrongSamples.log'), data=trainWrite}

   
   -- next iteration:
   confusion:zero()
end
