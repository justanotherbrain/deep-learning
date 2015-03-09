-----------------------------------------------------------------------
-- Now it's time to make our model
-- We will begin by building our unsupervised model. We will then go
-- on to train the supervised portion.
--
-- We will also extract more features via a V1 model.
-----------------------------------------------------------------------

require 'torch'
require 'image'
require 'cunn'
require 'nn'
require 'unsup'
require 'xlua'



if not opt then
	print '==> processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('ConvNN Model Definition')
	cmd:text()
	cmd:text('Options:')
	cmd:option('-comp_type', 'gpu', 'cpu | gpu')
	cmd:text()
	opt = cmd:parse(arg or {})
end

function RemoveLastLayers(model, n)
  --This assumes we're using nn.Sequential as out base...fix it if you want
  if n == 0 then return model end
  ret = nn.Sequential()
  for i = 1,model:size()-n do
    ret:add(model:get(i):clone())
  end
  return ret
end
------------------------------------------------------------------------
print '==> define parameters'

noutputs = 10 -- 10 classes

encoder = torch.load("encoder.net")
model = RemoveLastLayers(encoder,3)


-- input dimensions
nfeats = 3
width = 96
height = 96
ninputs = nfeats*width*height

-- hidden units, filter sizes, etc
nstates = {192, 192, 384}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)


if opt.comp_type == 'gpu' then
	--model = nn.Sequential()

	-- stage 1: filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(100, nstates[1], filtsize, filtsize))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	
	-- stage 2: filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	-- stage 3: standard 2-layer neural network
	model:add(nn.View(nstates[2]*filtsize*filtsize))
	model:add(nn.Dropout(0.5))
	model:add(nn.Linear(nstates[2]*filtsize*filtsize,nstates[3]))
	model:add(nn.ReLU())
	model:add(nn.Linear(nstates[3],noutputs))



else
	-- filter the images using a V1 filter (inspired by neuroscience). Then
	-- pus through a convolutional network.

	model = nn.Sequential()

	-- TODO:  First pass the V1 gabor filter through image to extract features.
	-- but for now, filter, squash, l2 pooling, normalize
	model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
	model:add(nn.Tanh())
	model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
	model:add(nn.SpatialSubtractiveNormalization(nstates[1],normkernel))

	-- layer 2: filter, squash l2 pooling, normalize
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
	model:add(nn.Tanh())
	model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
	model:add(nn.SpatialSubtractiveNormalization(nstates[2],normkernel))

	-- layer 3: standard 2-layer neural network
	
end

parameters,gradParameters = model:getParameters()

