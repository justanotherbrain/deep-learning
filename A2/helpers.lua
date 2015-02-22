function RemoveLastLayers(model, n)
  --This assumes we're using nn.Sequential as out base...fix it if you want
  if n == 0 then return model end
  ret = nn.Sequential()
  for i = 1,model:size()-n do
    ret:add(model:get(i):clone())
  end
  return ret
end
function CreateFolds(numModels, numSamples, folds)
  if folds == nil then
    folds = torch.randperm(numSamples)
  else
    if folds:size():size() == 1 then
      folds = torch.reshape(folds,folds:size()[1],1)
    end
    if folds:size(2) ~= 1 then
      print '==>Folds already in matrix form. Returning folds, assuming this is an appropriate fold matrix'
      return folds
    end
  end
  local n = math.floor(numSamples/numModels)
  folds = torch.reshape(folds[{{1,numModels * n}}], n, numModels)--To make this clean, ignore the last #samples % #models samples
  print ('==>Creating '.. numModels..' folds each with '.. n ..' samples')
  return folds
end
function CreateLogPackages(opt, parameters)
  local noutputs = parameters.noutputs
  ret = {}
  for i = 1,opt.models do
    ret[i] = {}
    if opt.trainSetOnly ~= 1 then 
      ret[i].testConfusion = optim.ConfusionMatrix(noutputs) 
      ret[i].testLogger = optim.Logger(paths.concat(opt.save, 'test' .. i .. '.log')) 
    end
    ret[i].trainConfusion = optim.ConfusionMatrix(noutputs)
    ret[i].trainLogger = optim.Logger(paths.concat(opt.save, 'train' .. i .. '.log'))
    ret[i].log = 
    function(self)
      if self.testLogger ~= nil then
        print '===>Test Confusion'
        print (self.testConfusion)
        
        self.testLogger:add{['% mean class accuracy (train set)'] = self.testConfusion.totalValid * 100,
            ['1'] = self.trainConfusion.valids[1],
            ['2'] = self.trainConfusion.valids[2],
            ['3'] = self.trainConfusion.valids[3],
            ['4'] = self.trainConfusion.valids[4],
            ['5'] = self.trainConfusion.valids[5],
            ['6'] = self.trainConfusion.valids[6],
            ['7'] = self.trainConfusion.valids[7],
            ['8'] = self.trainConfusion.valids[8],
            ['9'] = self.trainConfusion.valids[9],
            ['0'] = self.trainConfusion.valids[10]
          }
      end
      print '===>Train Confusion'
      print (self.trainConfusion)

      self.trainLogger:add{['% mean class accuracy (train set)'] = self.trainConfusion.totalValid * 100,
          ['1'] = self.trainConfusion.valids[1],
          ['2'] = self.trainConfusion.valids[2],
          ['3'] = self.trainConfusion.valids[3],
          ['4'] = self.trainConfusion.valids[4],
          ['5'] = self.trainConfusion.valids[5],
          ['6'] = self.trainConfusion.valids[6],
          ['7'] = self.trainConfusion.valids[7],
          ['8'] = self.trainConfusion.valids[8],
          ['9'] = self.trainConfusion.valids[9],
          ['0'] = self.trainConfusion.valids[10]
        }
    end
  end
  ret.logmodel = 
  function(filename, model) 
     os.execute('mkdir -p ' .. sys.dirname(filename))
     print('==> saving model to '..filename)
     torch.save(filename, model)
  end
  return ret
end

function LogParameters(opt, parameters)
  paramLogger = optim.Logger(paths.concat(opt.save, 'params.log'))


  paramLogger:add{['maxIter'] = opt.maxIter, ['momentum'] = opt.momentum,
   ['weightDecay'] = opt.weightDecay, ['model'] = opt.model, ['optimization'] = opt.optimization, ['learningRate'] = opt.learningRate, ['loss'] = opt.loss, ['batchSize'] = opt.batchSize}
end

function OptimizerAndCriterion(opt)
  --This only considers nll criterion as of now...so shoot me
  if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

  elseif opt.optimization == 'LBFGS' then
     optimState = {
        learningRate = opt.learningRate,
        maxIter = opt.maxIter,
        nCorrection = 10
     }
     optimMethod = optim.lbfgs

  elseif opt.optimization == 'SGD' then
     optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = 1e-7
     }
     optimMethod = optim.sgd

  elseif opt.optimization == 'ASGD' then
     optimState = {
        eta0 = opt.learningRate,
        t0 = trsize * opt.t0
     }
     optimMethod = optim.asgd

  else
     error('unknown optimization method')
  end
  if opt.loss == 'nll' then criterion = nn.ClassNLLCriterion() else error('nll only so far') end
  return {optimState=optimState, optimMethod=optimMethod, criterion=criterion}
end

