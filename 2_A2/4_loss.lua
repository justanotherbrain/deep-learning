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
	cmd:option('-loss','nll','type of loss function to minimize: nll, distance')
	cmd:text()
	opt = cmd:parse(arg or {})
	--model = nn.Sequential()
end

-- 10 classes
noutputs = 10

print '==> define loss'


--model:add(nn.LogSoftMax())

loss = 'nll'

if loss == 'nll' then
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()

elseif opt.loss == 'distance' then
	-- This will try to minimze the confusion matrix by pushing the confusion matrix to the identity matrix.
	criterion = nn.DistKLDivCriterion()
	-- the perfect distribution on the confusion matrix is the identity matrix
	idealMatrix = torch.eye(noutputs)


else
	error('unknown -loss')
end



print '==> the loss function: '
print(criterion)

