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












