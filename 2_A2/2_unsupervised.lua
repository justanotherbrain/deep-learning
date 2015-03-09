-----------------------------------------------------------------------
-- Unsupervised learning with unlabeled data set
-- The point of this will be to maximize information extraction
-- I should say that it is trying to maximize information given
-- some constraints about how the information should be represented.
-- 
--
-----------------------------------------------------------------------
require 'torch'
require 'image'
require 'unsup'
require 'nn'
require 'cunn'
require 'xlua'
require 'optim'

-----------------------------------------------------------------------

function getdata(datafile, inputsize, std)
	local data = datafile
	local dataset ={}
	local std = std or 0.2
	local nsamples = data:size(1)
	local nrows = data:size(3)
	local ncols = data:size(4)
	function dataset:size()
		return nsamples
	end
	function dataset:selectPatch(nr,nc)
		local imageok = false
		if simdata_verbose then
			print('selectPatch')
		end
		while not imageok do
			--image index
			local i = math.ceil(torch.uniform(1e-12,nsamples))
			local im = data:select(1,i)
			-- select some patch for original that contains original + pos
			local ri = math.ceil(torch.uniform(1e-12,nrows-nr))
			local ci = math.ceil(torch.uniform(1e-12,ncols-nc))
			local patch = im:narrow(2,ri,nr)
			patch = patch:narrow(3,ci,nc)
			local patchstd = patch:std()
			if data_verbose then
				print('Image ' .. i .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
			end
			if patchstd > std then
				if data_verbose then
					print(patch:min(),patch:max())
				end
				return patch,i,im
			end
		end
	end
	local dsample = torch.Tensor(inputsize*inputsize*3)
	function dataset:conv()
		dsample = torch.Tensor(1,inputsize,inputsize)
	end
	setmetatable(dataset, {__index = function(self, index)
				local sample,i,im = self:selectPatch(inputsize, inputsize)
				dsample:copy(sample)
				return {dsample,dsample,im}
				end})
	return dataset
end

---------------------------------------------------------------------

dataset = getdata(unlabeledData.data,96)

nout = 5000
nfeats = 3
width = 96
height = 96
ninputs = nfeats*width*height
nstates = {192,192,384}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)
nn.hessian.enable()

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolutionMM(ninputs,nstates[1],filtsize,filtsize))
encoder:add(nn.ReLU())
encoder:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

encoder:add(nn.Linear(nstates[1]*filtsize*filtsize,nout))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(nout))


lambda = 1
beta = 1
decoder = unsup.LinearFistaL1(ninputs,nout)

module = unsup.PSD(encoder, decoder, beta)

print '==> training model'

module:initDiagHessianParameters()
x,dl_dx,ddl_ddx = module:getParameters()

hessianinterval = 10000
local err = 0
local iter = 0
statiters = 5000
batchsize = 5
maxiters = 50000
for t = 1,maxiters,batchsize do

	if math.fmod(t,hessianinterval)==1 then
		local hessiansamples = 500 
		local minhessian = 0,02
		local maxhessian = 500
		local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
		etas = etas or ddl_ddx:clone()
	
		print('==> estimating diagonal hessian elements')
		for i = 1,hessiansamples do
			local ex = dataset[i]
			local input = ex[1]
			local target = ex[2]
			module:updateOutput(input,target)
		
			dl_dx:zero()
			module:updateGradInput(input,target)
			module:accGradParameters(input,target)
		
			ddl_ddx:zero()
			module:updateDiagHessianInput(input,target)
			module:accDiagHessianParameters(input,target)

			ddl_ddx_avg:add(1/hessiansamples,ddl_ddx)
		end
	
		print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
		ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
		ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
		print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
	
		etas:fill(1):cdiv(ddl_ddx_avg)
	end

	iter = iter+1
	xlua.progress(iter,statiters)
	
	local example = dataset[t]
	local inputs = {}
	local targets = {}
	for i = t,t+batchsize-1 do
		local sample = dataset[i]
		local input = sample[1]:clone()
		local target = sample[2]:clone()
		table.insert(inputs,input)
		table.insert(targets,target)
	end
	
	local feval = function()
		local f = 0
		dl_dx:zero()
	
		for i=1,#inputs do
			f=f+module:updateOutput(inputs[i],targets[i])
			module:updateGradInput(inputs[i],targets[i])
			module:accGradParameters(inputs[i],targets[i])
		end
	
		dl_dx:div(#inputs)
		f=f/#inputs
		return f,dl_dx
	end

	sdgconf = sdgconf or {learningRate = 2e-3,
				learningRateDecay = 1e-5,
				learningRates = etas,
				momentum = 0}
	_,fs=optim.sgd(feval,x,sgdconf)
	err = err + fs[1]

	module:normalize()

	if math.fmod(t,statiters)==0 then
		print('==> iteration = ' .. t .. ', average loss = ' .. err/statiter)
		err = 0; iter = 0
	end
end




