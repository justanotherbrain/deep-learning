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
	save = 'featureEncoder'
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

shuffled_data = torch.FloatTensor(surrogateData.size()[1],3,36,36)
shuffled_labels = torch.FloatTensor(surrogateData.size()[1])
shuffle = torch.randperm(surrogateData.size()[1])
for i = 1,shuffle:size()[1] do
	shuffled_data[i] = surrogateData.data[shuffle[i]]
	shuffled_labels[i] = surrogateData.labels[shuffle[i]]
end
surrogateData = {
	data = shuffled_data,
	labels = shuffled_labels,
	size = function() return loaded:size() end
}

for i = 1,surrogateData.data:size()[1] do
        surrogateData.data[i] = image.rgb2yuv(surrogateData.data[i])
end

channels = {'y','u','v'}

mean = {}
std = {}

for i,channel in ipairs(channels) do
        mean[i] = surrogateData.data[{ {}, i, {}, {} }]:mean()
        std[i] = surrogateData.data[{ {}, i, {}, {} }]:std()
        surrogateData.data[{ {}, i, {}, {} }]:add(-mean[i])
        surrogateData.data[{ {}, i, {}, {} }]:div(std[i])
end


neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1,neighborhood,1):float()

for c in ipairs(channels) do

	for i = 1,surrogateData.data:size()[1] do
                surrogateData.data[{i,{c},{},{}}]=normalization:forward(surrogateData.data[{i,{c},{},{}}])
        end
end

model,criterion = create_network(8000, surrogateData.size()[4])
for i = 1,100 do
	train_model(model,criterion,surrogateData)
end





