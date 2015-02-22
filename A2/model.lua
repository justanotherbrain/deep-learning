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
print(parameters)

function CreateModel(parameters, opt)
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
if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'convnet' then
   if opt.type == 'cuda' then
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
if opt.loss == 'nll' then model:add(nn.LogSoftMax()) end
return model
end
