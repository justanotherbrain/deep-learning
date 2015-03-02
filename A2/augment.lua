-------------------------------------------------------------------------
-------------------------------------------------------------------------
--																	   --
-- Sam Royston, Team Monkey Business, Spring 2015					   --
--																	   --
-- Functions for augmenting data via different transforms			   --
--																	   --
-- Default values taken from "Discriminative Unsupervised Feature 	   --
-- Learning with Convolutional Neural Networks" by Dosovitskiy et. al. --
--																	   --
-------------------------------------------------------------------------
-------------------------------------------------------------------------



require 'torch'
require 'image'
require 'xlua'
require 'math'


function generate_samples( input, commitee_count, real_seeds_count, faux_samples_count, patch_size )
	
	--
	-- this is to generate a set of aumented data
	-- commitee_count: number of disjoint sets of samples are used for aumentation
	-- real_seeds_count: number of random real images used to generate the new ones
	-- samples_count: number of faux samples to be generated from each real one 
	--
	-- note that real_seeds_count * commitee_count cannot exceed the input size, since we are subtractively sampling
	--


	commitee_count = commitee_count or 2
	real_seeds_count = real_seeds_count or 10
	shuffle_indices = torch.randperm(commitee_count * real_seeds_count)
	patch_size = patch_size or 36 
	faux_samples_count = faux_samples_count or 64
	progress = 0


	for i=1,commitee_count do
	 	local to_augment = torch.Tensor(real_seeds_count, input:size()[2], 96, 96)
	 	local labels = torch.Tensor(real_seeds_count * faux_samples_count)
		for j=1,real_seeds_count do
			local index = (commitee_count - 1) * real_seeds_count  + j
			to_augment[index % real_seeds_count + 1] = input[shuffle_indices[index]]
			labels:sub(index % real_seeds_count + 1, (index % real_seeds_count + 1) + faux_samples_count):fill(index) 
		end
        augmented_data = apply_trans(to_augment, patch_size, faux_samples_count)
        torch.save('SurrogateData/surrogate_traindata_' .. i .. '.t7', augmented_data)
        torch.save('SurrogateData/surrogate_labels_' .. i .. '.t7', labels)
	end
	print (faux_samples_count * real_seeds_count * commitee_count .. ' training images generated in ' .. commitee_count .. ' files')
end

function apply_trans( imgs ,patch_size , samples_count, transforms )

	-- conveinience method for using this packages tools  
	-- successively apply transforms to each image.     

	transforms = transforms or {flip_transform, rotate_transform, translate_transform, scale_transform, get_inset, color_transform}
	patch_size = patch_size or 36 
	samples_count = samples_count or 64
	
	-- only for visualization
	new_imgs = torch.Tensor(samples_count * imgs:size()[1], imgs:size()[2], patch_size, patch_size)
	for i = 1, imgs:size()[1] do
		xlua.progress(i, imgs:size()[1])	
		for j=1,samples_count do
			repeat 
				success = false
				im = imgs[i]
				for f_i = 1, table.getn(transforms) do
					f = transforms[f_i]
					success, im = pcall(f,im)
					if not success then break end
				end
			until success == true
		    new_imgs[(i - 1) * samples_count + j] = im
		end
	end
	return new_imgs
end


function get_inset( img, patch_size )
	
	-- take inner rectangle or "patch"
	
	start_dim = img:size()[3]
	patch_size = patch_size or 36
	border = math.floor(start_dim - patch_size) / 2
	new_img = torch.Tensor(3, patch_size, patch_size)
	new_img[1] = img[1]:sub(border, border + patch_size - 1, border, border + patch_size - 1)
	new_img[2] = img[2]:sub(border, border + patch_size - 1, border, border + patch_size - 1)
	new_img[3] = img[3]:sub(border, border + patch_size - 1, border, border + patch_size - 1)
	return new_img
end

function translate_transform( img, shift_wrt_patch, patch_size )
	
	-- applies random translation proportional to patch size

	start_dim = img:size()[3]
	patch_size = patch_size or 36
	shift_wrt_patch = shift_wrt_patch or 0.2 
	max_shift = shift_wrt_patch * patch_size
	r = (torch.rand(2) * max_shift) -  max_shift / 2
	return image.translate(img ,r[1], r[2])
end

function rotate_transform( img, max_theta )

	-- applies random rotation

	max_theta = max_theta or math.pi * 0.23
	r = (torch.rand(2) * max_theta) -  max_theta / 2
	return image.rotate(img ,r[1])		
end

function flip_transform( img )

	-- this should be ok for everthing except for text right?

	if torch.uniform() < 0.5 then
		return image.hflip(img)
	else
		return img
	end

end

function color_transform( img, max_hue_shift, max_sat_shift, max_val_shift )
	
	-- applies color tranform to images
	
	max_hue_shift = max_hue_shift or 0.1
	max_sat_shift = max_sat_shift or 0.5
	max_val_shift = max_val_shift or 0.6

	img = image.rgb2hsv(img)

	h =  1.0 + (torch.uniform() - 0.5) * 2 * max_hue_shift
	s = ((torch.uniform() - 0.5) * 2 * max_sat_shift) + 1
	v = ((torch.uniform() - 0.5) * 2 * max_val_shift) + 1


	hsvimg = torch.Tensor(img:size()[1],img:size()[2],img:size()[3])

	norm_vals = img[3] / 256.0 

	hsvimg[1] = img[1] + h
	hsvimg[2] = torch.pow(img[2],s)
	hsvimg[3] = torch.pow(norm_vals,v) * 256.0

	hsvimg = image.hsv2rgb(hsvimg)
	
	return hsvimg
end

function scale_transform( img, max_shift, lower_bound )
	
	-- applies random scaling proportional to patch size

	max_shift = max_shift or 0.5
	lower_bound = lower_bound or 0.75
	scale = torch.rand(2) * max_shift + lower_bound
	return image.scale(img, img:size()[2] * scale[1], img:size()[3] * scale[2])
end
