require 'csvigo'
require 'representation'

data_directory = 'LearnedFeatures'

data_directory = 'stl10_binary'

train_file = 'train_X.bin'
train_labels_file = 'train_y.bin'

test_file = 'test_X.bin'
test_labels_file = 'test_y.bin'




function generate_feats( model, test )
    
    -- generate features for data

    features = torch.Tensor(test:size()[1])
    for i=1, test.size()[1] do
        features[i] = model.forward(test[i])
    end
    return features
end

function get_feature_model(model, n)
    
    -- remove last layers of model

    if n == 0 then return model end
    ret = nn.Sequential()
    for i = 1,model:size()-n do
        ret:add(model:get(i):clone())
    end
    return ret
end 



training_label_fd = torch.DiskFile(data_directory .. train_labels_file, "r", true)
training_label_fd:binary():littleEndianEncoding()
training_labels = torch.ByteTensor(5000, 1)
training_label_fd:readByte(training_labels:storage())

training_data_fd = torch.DiskFile(data_directory .. train_file, "r", true)
training_data_fd:binary():littleEndianEncoding()
training_data = torch.ByteTensor(5000, 3, 96, 96)
training_data_fd:readByte(training_data:storage())

shuf_labels = torch.ByteTensor(5000, 1)
shuf_data = torch.ByteTensor(5000, 3, 96, 96)

shuffle = torch.randperm(training_data.size()[1])


for i=1,training_data.size()[1] do
	shuf_labels[i] = training_labels[shuffle[i]]
	shuf_data[i] = training_data[shuffle[i]]
end



train_x = shuf_data:sub(1 , 4000)
train_y = shuf_labels:sub(1 , 4000)

valid_x = shuf_data:sub(4001, 5000)
valid_y = shuf_labels:sub(4001, 5000)

-- load pre trained model

model = torch.load('featureEncoder.net')

model = get_feature_model(model, 5)

model:add(nn.View(nstates[2]*filtsize*filtsize))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))





-- folds = 5



-- for i=1,folds do
-- 	train = torch.Tensor(training_data:size()[1] - training_data:size()[1] / folds, 3, 36, 36)
-- 	train:sub(1, (i - 1) * (training_data:size()[1] / folds) + 1):fill()
-- 	validation = training_data:sub( (i - 1) * (training_data:size()[1] / folds) + 1 ) , i * (training_data:size()[1] / folds))
-- end
