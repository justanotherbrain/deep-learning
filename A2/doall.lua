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
  cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin | mmc. To train multiple models with different criterion as follows: nll/nll/mmc/mse...etc')

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
  cmd:option('-models', 2, 'number of models to train')
  cmd:option('-maxEpoch', 4, 'number of epochs to train for without seeing the best guess improve')
  cmd:option('-angle', math.pi/18, 'angle to rotate the training images')
  cmd:option('-hflip', 1, 'reflect training images? 1|0')
  cmd:option('-folds', 1, 'input to CreateFolds. <=1 for one model, otherwise, should equal #models')
  cmd:option('-trteb', 3, 'train test or both 1=train,2=test,3=both')
  cmd:option('-kaggle','kaggle.csv','where to save to')
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

function DoAll(opt)

  print('\n=>Loading data')
  dofile 'data.lua'  
  trainData, testData = ReadFiles(opt)
  TransformImages(trainData, opt)
  Preprocess(trainData, testData, opt)
  print('\n=>Loading needed files')
  dofile 'model.lua'
  dofile 'combine.lua'

  if opt.size ~= 'debug' and trainData.fold_indices ~= nil then 
    folds = trainData.fold_indices 
  else 
    folds = opt.folds
  end
  if opt.trteb ~= 2 then
    --Create models
    append(opt, parameters)
    if opt.models > 1 then
      local temp = parameters
      parameters = {noutputs=parameters.noutputs}
      for i = 1,opt.models do
        local copy = shallowcopy(temp)
        table.insert(parameters, copy)
      end
    end
    parseLoss = stringSplit(opt.loss, '/')
    if #parseLoss > 1 then
      if #parseLoss ~= opt.models then
        print 'INCORRECT NUMBER OF CRITERIA SELECTED'
        return 
      end
      for i = 1,#parseLoss do
        parameters[i].loss = parseLoss[i]
      end
    end

    model_optim_critList = CreateModels(opt, parameters, ModelOptimCrit)
    --Create LogPackages
    logpackages = CreateLogPackages(opt, parameters, opt.folds)
    --Train and combine models
    combined = TrainModels(model_optim_critList, opt, trainData, Train, folds, logpackages)
  end
  --Test the data
  if opt.trteb ~= 1 then 
    opt.noutputs = parameters.noutputs
    LoadAndTest(opt, testData,'combined_model.net', opt.kaggle) 
  end
 return model_optim_critList
end
print('=> Parsing command line')
opt = ParseCommandline()
print('Executing training and testing proccedure')
DoAll(opt)
