---------------------------------------------------------------------
-- rabadi
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128,256}
filtsize = 3
poolsize = 2
poolsize2 = 2
normkernel = image.gaussian1D(7)
----------------------------------------------------------------------
print '==> construct model'

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

-- stage 3 : 
model:add(nn.SpatialConvolutionMM(nstates[2],nstates[3],filtsize,filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[3],2,poolsize2,poolsize2,poolsize2,poolsize2))
model:add(nn.SpatialSubtractiveNormalization(nstates[3], normkernel))


-- stage 3 : standard 2-layer neural network
--local tmp = filtsize
local tmp = 2
model:add(nn.Reshape(nstates[3]*tmp*tmp))
model:add(nn.Linear(nstates[3]*tmp*tmp, nstates[4]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[4], noutputs))


----------------------------------------------------------------------
print '==> here is the model:'
print(model)
----------------------------------------------------------------------
---- Visualization is quite easy, using gfx.image().
--
