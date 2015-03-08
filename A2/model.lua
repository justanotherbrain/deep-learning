require 'torch'   -- torch
require 'image'   -- for image transforms
--require 'cunn'      -- provides all sorts of trainable modules/layers
require 'nn'      -- provides all sorts of trainable modules/layers

parameters = {
  -- 10-class problem
noutputs = 10,

-- input dimensions
nfeats = 3,
width = 96,
height = 96,

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128},
filtsize = 5,
poolsize = 2,
normkernel = image.gaussian1D(7),
layersToRemove = 0, addSoftMax = false
}
print '==> define parameters'

--Create a model, an optimizer, and a criterion
function ModelOptimCrit(parameters)
 local noutputs = parameters.noutputs
 local nfeats = parameters.nfeats
 local width = parameters.width
 local height = parameters.height
 local nstates = parameters.nstates
 local filtsize = parameters.filtsize
 local poolsize = parameters.poolsize
 local normkernel = parameters.normkernel
  ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2
if parameters.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif parameters.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

elseif parameters.model == 'convnet' then
   if parameters.type == 'cuda' then
      -- a typical modern convolution network (conv+relu+pool)
      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      dim = ((width - filtsize + 1)/poolsize - filtsize + 1)/poolsize
      -- stage 3 : standard 2-layer neural network
      model:add(nn.View(nstates[2]*dim*dim))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[2]*dim*dim, nstates[3]))
      model:add(nn.ReLU())
      model:add(nn.Linear(nstates[3], noutputs))

   else
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      dim = ((width - filtsize + 1)/poolsize - filtsize + 1)/poolsize
      model:add(nn.Reshape(nstates[2]*dim*dim))
      model:add(nn.Linear(nstates[2]*dim*dim, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))
   end
else
   error('unknown -model')
end
if parameters.loss == 'nll' or parameters.loss == 'dkld' then model:add(nn.LogSoftMax()) end
ret = OptimizerAndCriterion(parameters)
ret.model = model
return ret
end

function Train(model_optim_crit, trainData, opt, confusion, indicies)
  local optimMethod = model_optim_crit.optimMethod
  local optimState = model_optim_crit.optimState
  local criterion
  local model
  if opt.type == 'cuda' then    
    criterion = model_optim_crit.criterion:cuda()
    model = model_optim_crit.model:cuda()
  else
    criterion = model_optim_crit.criterion:double()
    model = model_optim_crit.model:double()
  end
  
  ret = {err=0}
  model:training()
  shuffle = torch.randperm(indicies:size(1))
  local parameters,gradParameters = model:getParameters()
  for t = 1,indicies:size(1),opt.batchSize do
    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,indicies:size(1)) do
       -- load new sample
       local input = trainData.X[indicies[shuffle[i]]]
       local target = trainData.y[indicies[shuffle[i]]]
       if opt.type == 'double' then input = input:double()
       elseif opt.type == 'cuda' then input = input:cuda() end
       table.insert(inputs, {input, indicies[i]})--DG change, Inputs is now an array of tuples
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
                        local err = criterion:forward(output, targets[i][1])
                        f = f + err

                        -- estimate df/dW
                        local df_do = criterion:backward(output, targets[i][1])
                        model:backward(inputs[i][1], df_do)
                     end
                     -- normalize gradients and f(X)
                     gradParameters:div(#inputs)
                     f = f/#inputs
                     
                     -- return f and df/dX
                     return f,gradParameters
                  end

    -- optimize on current mini-batch
    if optimMethod == optim.asgd then
       _,_ ,average = optimMethod(feval, parameters, optimState)
    else
       optimMethod(feval, parameters, optimState)
    end   
    -- log if samples are wrong. DG addition
    local _, guess  = torch.max(model.output,1)
    if confusion ~= nil then confusion:add(model.output, targets[1][1]) end
    ret.err = ret.err + ((guess[1] ~= targets[1][1]) and 1 or 0)
 end
  if opt.type == 'cuda' or opt.type == 'float' then
    criterion = model_optim_crit.criterion:float()
    model = model_optim_crit.model:float()
  end
 ret.err = ret.err/indicies:size(1)
 return ret
end

function OptimizerAndCriterion(parameters)
  --This only considers nll criterion as of now...so shoot me
  if parameters.optimization == 'CG' then
   optimState = {
      maxIter = parameters.maxIter
   }
   optimMethod = optim.cg

  elseif parameters.optimization == 'LBFGS' then
     optimState = {
        learningRate = parameters.learningRate,
        maxIter = parameters.maxIter,
        nCorrection = 10
     }
     optimMethod = optim.lbfgs

  elseif parameters.optimization == 'SGD' then
     optimState = {
        learningRate = parameters.learningRate,
        weightDecay = parameters.weightDecay,
        momentum = parameters.momentum,
        learningRateDecay = 1e-7
     }
     optimMethod = optim.sgd

  elseif parameters.optimization == 'ASGD' then
     optimState = {
        eta0 = parameters.learningRate,
        t0 = trsize * parameters.t0
     }
     optimMethod = optim.asgd

  else
     error('unknown optimization method')
  end
  if parameters.loss == 'nll' then 
    criterion = nn.ClassNLLCriterion()
  elseif parameters.loss == 'dkld' then
    criterion = nn.DistKLDivCriterion()
  elseif parameters.loss == 'mmc' then
    criterion = nn.MultiMarginCriterion(2)
  else 
    error('nll/dkld/mmc only so far') 
  end
  return {optimState=optimState, optimMethod=optimMethod, criterion=criterion} 
end
function CreateLogPackages(opt, parameters, numFolds)
  ret = {}
  for i = 1,opt.models do
    ret[i] = {}
    if numFolds ~= 1 then 
      ret[i].testConfusion = optim.ConfusionMatrix(parameters[i].noutputs) 
      ret[i].testLogger = optim.Logger(paths.concat(opt.save, 'test' .. i .. '.log')) 
    end
    ret[i].trainConfusion = optim.ConfusionMatrix(parameters[i].noutputs)
    ret[i].trainLogger = optim.Logger(paths.concat(opt.save, 'train' .. i .. '.log'))
    ret[i].log = 
    function(self)
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
      if self.testLogger ~= nil then
        print '===>Test Confusion'
        print (self.testConfusion)
        
        self.testLogger:add{['% mean class accuracy (validation set)'] = self.testConfusion.totalValid * 100,
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
  end
  return ret
end

function LogParameters(opt, parameters)
  paramLogger = optim.Logger(paths.concat(opt.save, 'params.log'))


  paramLogger:add{['maxIter'] = parameters.maxIter, ['momentum'] = parameters.momentum,
   ['weightDecay'] = parameters.weightDecay, ['model'] = parameters.model, ['optimization'] = parameters.optimization, ['learningRate'] = parameters.learningRate, ['loss'] = parameters.loss, ['batchSize'] = parameters.batchSize}
end
