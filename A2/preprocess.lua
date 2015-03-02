require "torch"
require "image"
require "xlua"

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

function load_train_data()
	
	-- load from file
	
	data_filename, label_filename = train_decorate(1)
	data = torch.load(data_filename)
	labels = torch.load(label_filename)
	return data, labels
end