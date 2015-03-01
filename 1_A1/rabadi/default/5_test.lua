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

   
   -- local vars
   local time = sys.clock()

   -- top score to save corresponding to saved model
   top_score = top_score or 0.1


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
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot ... along with accuracy scores for each digit
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100, 
      ['1'] = confusion.valids[1],
      ['2'] = confusion.valids[2],
      ['3'] = confusion.valids[3],
      ['4'] = confusion.valids[4],
      ['5'] = confusion.valids[5],
      ['6'] = confusion.valids[6],
      ['7'] = confusion.valids[7],
      ['8'] = confusion.valids[8],
      ['9'] = confusion.valids[9],
      ['0'] = confusion.valids[10]
   }
   
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- here we check to see if the current model is the best yet, and if so, save it
   if top_score < confusion.totalValid * 100 then
      local top_filename = paths.concat(opt.save, 'winning_model.net')
      os.execute('mkdir -p ' .. sys.dirname(top_filename))
      print('==> saving new top model to '..top_filename)
      torch.save(top_filename, model)
      top_score = confusion.totalValid * 100
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
