require 'torch'
require 'nn'
require 'image'


opt = {
	type = 'cuda',
	batchSize = 5
}



require 'representation.lua'
i = 1


surrogate_file_labels = 'SurrogateData/surrogate_labels_' .. i .. '.t7'
surrogate_file_data = 'SurrogateData/surrogate_traindata_' .. i .. '.t7'

loaded = torch.load(surrogate_file_data)
ls = torch.load(surrogate_file_labels) 
surrogateData = {
	data = loaded,
	labels = ls,
	size = function() return loaded:size() end
}


model,criterion = create_network(4000, 36)

for i = 1,100 do
	train_model(model,criterion,surrogateData)
end


