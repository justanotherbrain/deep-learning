-----------------------------------------------------------------
-- Monkeybusiness
-- 
-- This script defines the loss function
--
-----------------------------------------------------------------

require 'torch'
require 'nn'

if not opt then
	print '==> processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Pick loss function')
	cmd:text()
	cmd:text('Options:')
	cmd:option('-loss','distance','type of loss function to minimize: nll, distance')
	cmd:text()
	opt = cmd:parse(arg or {})
	model = nn.Seuential()
end

-- 10 classes
noutputs = 10

print '==> define loss'


model:add(nn.LogSoftMax())

if opt.loss == 'nll' then
	criterion = nn.ClassNLLCriterion()

elseif opt.loss = 'distance' then
	-- This will try to minimze the confusion matrix by pushing the confusion matrix to the identity matrix.
	criterion = nn.DistKLDivCriterion()
	-- the perfect distribution on the confusion matrix is the identity matrix
	idealMatrix = torch.eye(noutputs)


else
	error('unknown -loss')
end




