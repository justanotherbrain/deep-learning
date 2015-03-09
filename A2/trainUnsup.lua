require 'torch'
require 'nn'
require 'image'


opt = {
	learningRate = 1e-3,
	optimization = 'SGD',
	momentum = 0,
	weightDecay = 0,
	t0 = 1,
	type = 'cuda',
	batchSize = 5,
	save = 'featureEncoder.net'
}



--require 'representation.lua'
dofile('representation.lua')

i = 1

surrogate_file_labels = 'SurrogateData/surrogate_labels_' .. i .. '.t7'
surrogate_file_data = 'SurrogateData/surrogate_traindata_' .. i .. '.t7'

ll = torch.Tensor(8000*16)
n = 0
for i = 0,ll:size()[1]-1 do
	if i%16 == 0 then
		n = n+1
	end
	ll[i+1] = n
end

loaded = torch.load(surrogate_file_data)
ls = torch.load(surrogate_file_labels) 
surrogateData = {
	data = loaded,
	labels = ll,
	size = function() return loaded:size() end
}


model,criterion = create_network(8000, surrogateData.size()[4])

for i = 1,100 do
	train_model(model,criterion,surrogateData)
end





