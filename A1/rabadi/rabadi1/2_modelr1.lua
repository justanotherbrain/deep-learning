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
nstates = {64,64,128,128,256}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)
normkernel2 = image.gaussian1D(7)
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
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel2))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel2))

-- stage 3 : 
model:add(nn.SpatialConvolutionMM(nstates[2],nstates[3],filtsize,filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[3],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[3], normkernel2))

-- stage 4 :
model:add(nn.SpatialConvolutionMM(nstates[3],nstates[4],filtsize,filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[4],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[4], normkernel2))

-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(nstates[4]*filtsize*filtsize))
model:add(nn.Linear(nstates[4]*filtsize*filtsize, nstates[5]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[5], noutputs))


----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
---- Visualization is quite easy, using gfx.image().
--
if opt.visualize then
  if opt.model == 'convnet' then
     print '==> visualizing ConvNet filters'
     gfx.image(model:get(1).weight, {zoom=2, legend='L1'})
     gfx.image(model:get(5).weight, {zoom=2, legend='L2'})
  end
end
