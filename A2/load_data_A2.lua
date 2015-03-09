require 'torch'
require 'nn'
require 'image'
require 'xlua'
require 'augment'
require 'math'
require 'sys'

local matio = require 'matio'

if arg[1] == 'remote' then 
	data_directory = '../stl10_binary/'
	
else
	data_directory = '/Users/sam.royston/Downloads/stl10_binary/'
end

train_file = 'train_X.bin'
train_labels_file = 'train_y.bin'

test_file = 'test_X.bin'
test_labels_file = 'test_y.bin'

unlabeled_file = 'unlabeled_X.bin'


print('==> loading data from ' .. data_directory .. train_file)

-- Open the files and set little endian encoding

-- test data and labels
-- test_data_fd = torch.DiskFile(data_directory .. test_file, "r", true)
-- test_data_fd:binary():littleEndianEncoding()

-- test_label_fd = torch.DiskFile(data_directory .. test_labels_file, "r", true)
-- test_label_fd:binary():littleEndianEncoding()

-- -- training data and labels
-- training_data_fd = torch.DiskFile(data_directory .. train_file, "r", true)
-- training_data_fd:binary():littleEndianEncoding()

-- training_label_fd = torch.DiskFile(data_directory .. train_labels_file, "r", true)
-- training_label_fd:binary():littleEndianEncoding()

-- unlabeled data
unlabeled_data_fd = torch.DiskFile(data_directory .. unlabeled_file, "r", true)
unlabeled_data_fd:binary():littleEndianEncoding()

-- Create and read the data
-- test_data = torch.ByteTensor(8000, 3, 96, 96)
-- test_data_fd:readByte(test_data:storage())


-- test_labels = torch.ByteTensor(5000)
-- test_label_fd:readByte(labels:storage())

-- training_data = torch.ByteTensor(5000, 3, 96, 96)
-- test_data_fd:readByte(training_data:storage())

-- labels = torch.ByteTensor(5000)
-- label_fd:readByte(labels:storage())

unlabeled_data = torch.ByteTensor(100000, 3, 96, 96)
unlabeled_data_fd:readByte(unlabeled_data:storage())


-- Because data is in column-major, transposing the last 2 dimensions gives result that can be correctly visualized
unlabeled_data = unlabeled_data:transpose(3, 4)
-- test_data = test_data:transpose(3,4)

-- print('==> loading data from ' .. data_directory .. test_file)
-- test_data = matio.load(data_directory .. train_file)

-- print('==> loading data from ' .. data_directory .. unlabeled_file)
-- unlabeled_data = matio.load(data_directory .. unlabeled_file)



generate_samples(unlabeled_data, 10, 4000, 16)







-- for i=1,100 do

-- 	rotated = rotate_transform(single:reshape(3,96,96):transpose(3,2))
-- 	translated = translate_transform(rotated)
-- 	t[i] = color_transform(get_inset(translated))

-- 	-- t[i] = image.hsl2bgr(t[i])
-- 	-- t[i] = single
-- end

-- input = image.toDisplayTensor{
--    input=new_imgs, padding=3, nrow=32, saturate=false
-- }
-- torch.save('todraw.t7', input)


-- print(training_data["X"])

