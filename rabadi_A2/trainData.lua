require 'csvigo'
require 'representation'
require 'augment'
require 'torch'
require 'nn'
require 'cunn'
require 'xlua'

shuf_labels = torch.Tensor(5000, 1)
shuf_data = torch.Tensor(5000, 3, 36, 36)

shuffle = torch.randperm(training_data:size()[1])

for i=1,training_data:size()[1] do
        shuf_labels[i] = training_labels[shuffle[i]]
        shuf_data[i] = training_data[shuffle[i]]
end

train_x = shuf_data:sub(1 , 4000)
train_y = shuf_labels:sub(1 , 4000)

trainData = {
	data = train_x,
	labels = train_y,
	size = function() return train_x:size() end
}


valid_x = shuf_data:sub(4001, 5000)
valid_y = shuf_labels:sub(4001, 5000)

validData = {
	data = valid_x,
	labels = valid_y,
	size = function() return valid_x:size() end
}



