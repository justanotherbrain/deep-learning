require 'torch'
print '==> processing options'

-- current session's storage directory
dir_name = os.date():gsub(' ','_') .. ''

function ParseCommandline()
    
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('SVHN Loss Function')
  cmd:text()
  cmd:text('Options:')
  -- global:
  cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
  cmd:option('-threads', 6, 'number of threads')
  -- data:
  cmd:option('-size', 'debug', 'how many samples do we load: small | full | extra | debug')
  -- model:
  cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
  -- loss:
  cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')

  -- training:
  -- opt.save is where everything is saved

  cmd:option('-save', 'experiments/' .. dir_name .. '-Results', 'subdirectory to save/log experiments in')
  -- training:
  cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
  cmd:option('-plot', false, 'live plot')
  cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
  cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
  cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
  cmd:option('-momentum', 0, 'momentum (SGD only)')
  cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
  cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
  cmd:option('-type', 'double', 'type: double | float | cuda')
  cmd:option('-models', 1, 'number of models to train')
  cmd:option('-trainSetOnly', 0, 'do not use validation set for training on a given fold? 1|0')
  cmd:option('-maxEpoch', 4, 'number of epochs to train for without seeing the best guess improve')
  cmd:option('-angle', math.pi/18, 'angle to rotate the training images')
  cmd:option('-hflip', 1, 'reflect training images? 1|0')
  cmd:text()
  --DG addition
  opt = cmd:parse(arg or {})

  -- nb of threads and fixed seed (for repeatable experiments)
  if opt.type == 'float' then
     print('==> switching to floats')
     torch.setdefaulttensortype('torch.FloatTensor')
  elseif opt.type == 'cuda' then
     print('==> switching to CUDA')
     require 'cunn'
     torch.setdefaulttensortype('torch.FloatTensor')
  end
  torch.setnumthreads(opt.threads)
  torch.manualSeed(opt.seed)
  return opt

end

function DoAll()
  if false then 
    testData = torch.load('testData.t7')
    trainData = torch.load('trainData.t7')
  else
    dofile 'data.lua'  
  end
  dofile 'model.lua'
  dofile 'combine.lua'

  if opt.size ~= 'debug' then folds = trainData.fold_indices end
  --Create models
  model_optim_critList = CreateModels(opt, parameters, ModelOptimCrit)
  --Create LogPackages
  logpackages = CreateLogPackages(opt, parameters)
  --Train and combine models
  combined = TrainModels(model_optim_critList, opt, trainData, Train, nil, logpackages)
  --Test the data
  testCM = optim.Confusionmatrix(opt.noutputs)
  testResults = Test(combined, testData, opt, testCM)
  testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
  testLogger:add{['% mean class accuracy (test set)'] = testCM.totalValid * 100,
        ['1'] = testCM.valids[1],
        ['2'] = testCM.valids[2],
        ['3'] = testCM.valids[3],
        ['4'] = testCM.valids[4],
        ['5'] = testCM.valids[5],
        ['6'] = testCM.valids[6],
        ['7'] = testCM.valids[7],
        ['8'] = testCM.valids[8],
        ['9'] = testCM.valids[9],
        ['0'] = testCM.valids[10]
      }
end
print '==> executing all'
opt = ParseCommandline()
DoAll()
