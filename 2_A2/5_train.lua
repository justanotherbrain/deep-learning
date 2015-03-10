----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
require 'cunn'
----------------------------------------------------------------------
function getdata(datafile, inputsize, std)
        local data = datafile
        local dataset ={}
        local std = std or 0.2
        local nsamples = data:size(1)
        local nrows = data:size(3)
        local ncols = data:size(4)
        function dataset:size()
                return nsamples
        end
        function dataset:selectPatch(nr,nc)
                local imageok = false
                if simdata_verbose then
                        print('selectPatch')
                end
                while not imageok do
                        --image index
                        local i = math.ceil(torch.uniform(1e-12,nsamples))
                        local im = data:select(1,i)
                        -- select some patch for original that contains original + pos
                        local ri = math.ceil(torch.uniform(1e-12,nrows-nr))
                        local ci = math.ceil(torch.uniform(1e-12,ncols-nc))
                        local patch = im:narrow(2,ri,nr)
                        patch = patch:narrow(3,ci,nc)
                        local patchstd = patch:std()
                        if data_verbose then
                                print('Image ' .. i .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
                        end
                        if patchstd > std then
                                if data_verbose then
                                        print(patch:min(),patch:max())
                                end
                                return patch,i,im
                        end
                end
        end
        local dsample = torch.Tensor(inputsize*inputsize*3)
        function dataset:conv()
                dsample = torch.Tensor(1,inputsize,inputsize)
        end
        setmetatable(dataset, {__index = function(self, index)
                                local sample,i,im = self:selectPatch(inputsize, inputsize)
                                dsample:copy(sample)
                                return {dsample,dsample,im}
                                end})
        return dataset
end


trainDataset = getdata(unlabeledData.data,96)

----------------------------------------------------------------------

model:cuda()
criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files


trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
paramLogger = optim.Logger(paths.concat(opt.save, 'params.log'))


paramLogger:add{['maxIter'] = opt.maxIter, ['momentum'] = opt.momentum,
 ['weightDecay'] = opt.weightDecay, ['model'] = opt.model, ['optimization'] = opt.optimization, ['learningRate'] = opt.learningRate, ['loss'] = opt.loss, ['batchSize'] = opt.batchSize}

--if model then
--   parameters,gradParameters = model:getParameters()
--end

----------------------------------------------------------------------
print '==> configuring optimizer'

   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'


function train()
  
   -- epoch tracker
   epoch = epoch or 1
   
   
   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainDataset[shuffle[i]][1]
         local target = trainData.labels[shuffle[i]]
         input = input:cuda()
         table.insert(inputs, {input, shuffle[i]})--DG change, Inputs is now an array of tuples
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i][1])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i][1], df_do)
                          
                          -- log if samples are wrong. DG addition
                          _, guess  = torch.max(output,1)
                          trainSampleWrong[inputs[i][2]] = trainSampleWrong[inputs[i][2]] + ((guess[1] ~= targets[i]) and 1 or 0)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
         optimMethod(feval, parameters, optimState)
      
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)


   -- update logger/plot ... along with accuracy scores for each digit
     trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100,
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

      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
   
   trainWrite = {}
   for key, value in pairs(trainSampleWrong) do
      table.insert(trainWrite, {key,value})
   end
   table.insert(trainWrite, {'Epoch',#trainWrite+1})   
   table.sort(trainWrite, sampleComparer)
   trainWrite[1][2] = epoch - 1
   csvigo.save{path=paths.concat(opt.save, 'train_wrongSamples.log'), data=trainWrite}
  
  
   -- next epoch
   confusion:zero()
   epoch = epoch + 1

end

