## Architectures ##

### Defualt ###
Since the convolutional network was the best, I will go over 
the basics of that one.

There are locally-normalized hidden units and L2-Pooling (P=2).

Start by initializing the model:
-- STAGE1 --
1. add spatial convolution with 3 features (number of incoming planes - image is rgb), 64 states are then spit out, the convolution window is 5x5 - step defaults to 1.
2. Then add a non-linearity (tanh)
3. add a layer of lp-pooling - expect 64 input states, use 2 as pnorm, convolve over 2x2 with step sizes of 2x2
4. then spatial subtraction over size k for kernel (not exactly sure what numb er is here) of a neighborhood. This neighborhood is the topological definition (I think the kernel is basically a distance epsilon over all of the featuers. However, this makes me think that we could construct our own metric spaces, but what would the kernel be?
	- in this case, i think the kernel is a gaussian.
-- STAGE2 --
1. Add another spatial convolution, expecting 64 input states and extracting 64 output states. Convolve over 5x5 window with step size of 1.
2. Apply the non-linearity (tanh)
3. Repeat that spatial LP pooling with 64 input states, P=2, and convolve over 2x2 with 2x2 step.
4. Renormalize witht hat spatial subtraction with the same gaussian kernel.

-- STAGE3 (standard 2-layer neural network part) --
1. Reshape the matrix - 64*5*5.... That number sounds like magic to me.
2. Basically, it just puts all the elements out into a row vector
3. Now apply a linear transformation to 64*5*5 vector and push out some 128 element output
4. apply non-linearity
5. apply a second linear transformation that takes those 128 states and spits out 10 possible outputs.


-------------

### rabadi1 ###

This network is based off of the idea of, why the heck are we only using 3 hidden layers? This is the 21st century and I will not stand for anything less than 5 layers. Does the human brain have 5 layers of computation? Heck no!

Initialize the model:

-- STAGE1 --
1. Spatial convolution takes in 3 features





