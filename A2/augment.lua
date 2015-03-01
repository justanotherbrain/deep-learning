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

function apply_trans( imgs ,patch_size , samples_count, transforms )

	-- conveinience method for using this packages tools
	-- successively apply transforms to each image. 
	
	transforms = transforms or {rotate_transform, translate_transform, scale_transform, get_inset, color_transform}
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
