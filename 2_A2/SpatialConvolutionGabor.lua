local SpatialConvolutionGabor, parent = torch.class('SpatialConvolutionGabor', 'nn.Module')

function SpatialConvolutionGabor:__init(nInputPlane, nOutputPlane, kW, kH, lambda, theta, padding)
	parent.__init(self)
	self.nInputPlane = nInputPlane
	self.nOutputPlane = nOutputPlane
	self.kW = kW
	self.kH = kH
	self.lambda = lambda
	self.theta = theta
	self.sigma = 0.56*lambda
	self.Gabor = torch.Tensor(Sx,Sy)
	self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
	self.bias = torch.Tensor(nOutputPlane)
	self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
	self.gradBias = torch.Tensor(nOutputPlane)
	self.finput = torch.Tensor()
	self.fgradInput = torch.Tensor()
	self:reset()
end


function SpatialConvolutionMM:reset(stdv)
	if stdv then
		stdv = stdv * math.sqrt(3)
	else
		stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
	end
	if nn.oldSeed then
		self.weight:apply(function()
			return torch.uniform(-stdv, stdv)
		end)
		self.bias:apply(function()
			return torch.uniform(-stdv, stdv)
		end)
	else
		self.weight:uniform(-stdv, stdv)
		self.bias:uniform(-stdv, stdv)
	end
end


local function makeContiguous(self, input, gradOutput)
	if not input:isContiguous() then
		self._input = self._input or input.new()
		self._input:resizeAs(input):copy(input)
		input = self._input
	end
	if gradOutput then
		if not gradOutput:isContiguous() then
			self._gradOutput = self._gradOutput or gradOutput.new()
			self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
			gradOutput = self._gradOutput
		end
	end
	return input, gradOutput
end


local function GaborLayer
function SpatialConvolutionGabor:updateOutput(input)
	input = makeContiguous(self, input)
	return input.nn.SpatialConvolutionMM_updateOutput(self, input)
end


function SpatialConvolutionMM:updateGradInput(input, gradOutput)
	if self.gradInput then
		input, gradOutput = makeContiguous(self, input, gradOutput)
		return input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
	end
end



function SpatialConvolutionMM:accGradParameters(input, gradOutput, scale)
	input, gradOutput = makeContiguous(self, input, gradOutput)
	return input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
end



