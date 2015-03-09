require "torch"
require "image"
require "xlua"
require "math"

DEFAULT_DIR = "SurrogateData/"
TRAIN_PREFIX = "surrogate_taindata_"
LABELS_PREFIX = "surrogate_labels_"

function train_decorate( i )
	
	-- form the ith filenames

	return (DEFAULT_DIR .. TRAIN_PREFIX .. i .. ".t7"), (DEFAULT_DIR .. LABELS_PREFIX .. i .. ".t7") 
end

function convert_to_YUV( t )

	-- convert the compound tensor t, a set of images with expected dimension i x 3 x 36 x 36
	
	print ("converting to YUV -->")
	for i=1,t:size() do
		t[i] = image.rgb2yuv(t[i])
		xlua.progress(i, trainData:size())
	end
	return t
end

function load_data()
	
	-- load training data and labels from file
	
	data_filename, label_filename = train_decorate(1)
	data = convert_to_YUV(torch.load(data_filename))
	labels = torch.load(label_filename)
	return data, labels
end

function validation_split( data, fraction )
	
	-- separate data into train/test split

	split_index = math.floor(data:size() * fraction)
	train = data:sub(1,split_index)
	test = data:sub(split_index, data:size())
	
end


