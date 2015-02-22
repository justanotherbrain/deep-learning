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
  for i = 1,opt.numModels do
    ret[i] = {}
    ret[i].trainConfusion = optim.ConfusionMatrix(noutputs)
    ret[i].testConfusion = optim.ConfusionMatrix(noutputs)
    ret[i].trainLogger = optim.Logger(paths.concat(opt.save, 'train' .. i .. '.log'))
    ret[i].testLogger = optim.Logger(paths.concat(opt.save, 'test' .. i .. '.log'))
  end
  ret.filename = paths.concat(opt.save, 'model.net')
  ret.logmodel = function(filename, model) 
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
  end
  ret.log = function(self)
    for i = 1,#self do
      
      self[i].testLogger:add{['% mean class accuracy (train set)'] = .self[i].traintrainConfusion.totalValid * 100,
          ['1'] = self[i].trainConfusion.valids[1],
          ['2'] = self[i].trainConfusion.valids[2],
          ['3'] = self[i].trainConfusion.valids[3],
          ['4'] = self[i].trainConfusion.valids[4],
          ['5'] = self[i].trainConfusion.valids[5],
          ['6'] = self[i].trainConfusion.valids[6],
          ['7'] = self[i].trainConfusion.valids[7],
          ['8'] = self[i].trainConfusion.valids[8],
          ['9'] = self[i].trainConfusion.valids[9],
          ['0'] = self[i].trainConfusion.valids[10]
        }
      self[i].trainLogger:add{['% mean class accuracy (train set)'] = .self[i].traintestConfusion.totalValid * 100,
        ['1'] = self[i].testConfusion.valids[1],
        ['2'] = self[i].testConfusion.valids[2],
        ['3'] = self[i].testConfusion.valids[3],
        ['4'] = self[i].testConfusion.valids[4],
        ['5'] = self[i].testConfusion.valids[5],
        ['6'] = self[i].testConfusion.valids[6],
        ['7'] = self[i].testConfusion.valids[7],
        ['8'] = self[i].testConfusion.valids[8],
        ['9'] = self[i].testConfusion.valids[9],
        ['0'] = self[i].testConfusion.valids[10]
      }
    end
  end
  return ret
end

function LogParameters(parameters)
  
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

