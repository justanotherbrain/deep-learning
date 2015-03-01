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
	cmd:option('-comp_type', 'cpu', 'cpu | gpu')
	cmd:text()
	opt = cmd:parse(arg or {})
end

-----------------------------------------------------------------------
-- Gabor function as inspired by Prashant Lalwani
-- lambda is wavelength of filter
-- theta is orietnation of the gabor function (in degrees)
-- shi is phase offset
-- sigma is gaussian envelpoe
-- gamma is spatial aspect ratio

pi = 3.14
gamma = 0.5
shi = 0

function GaborLayer(Sx,Sy,lambda,theta)
        sigma = 0.56*lambda
        Gabor = torch.Tensor(Sx,Sy)
        for x = 1,Sx do
                for y = 1,Sy do
                       xPrime = (x-Sx/2-1)*math.cos(theta) + (y-Sy/2-1)*math.sin(theta) --equation 1
                        yPrime = -(x-Sx/2-1)*math.sin(theta) + (y-Sy/2-1)*math.cos(theta) --equation 2
                        Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime^2)+(yPrime^2 * gamma^2 )))*math.cos(2*pi*xPrime/lambda + shi) -- equation 3
                end
        end
        return(Gabor)
end



------------------------------------------------------------------------
print '==> define parameters'

noutputs = 10 -- 10 classes

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


if opt.type == 'gpu' then
	model = nn.Sequential()

	-- stage 1: filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	
	-- stage 2: filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
	model:add(nn.ReLu())
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
	






