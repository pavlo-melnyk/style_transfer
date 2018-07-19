''' The basic outline is this:
1. We are going to grab an image to reproduce its content.
2. We are going to create a VGG network, that stops at an intermediate convolution.
3. We are going to build up the loss function and the gradient, and then use
   a Scipy optimizer to solve for the ideal input image.
'''

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import keras.backend as K
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import fmin_l_bfgs_b

from datetime import datetime


def VGG16_AvgPool(shape):
	'''Changes all MaxPool layers to the AvgPool ones.
	Motivation: MaxPool throws away more information.
	'''
	vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
	my_model = Sequential()
	for layer in vgg.layers:
		if layer.__class__ == MaxPooling2D:
			my_model.add(AveragePooling2D())
		else:
			my_model.add(layer)
	return my_model


def VGG16_AvgPool_CutOff(shape, num_convs):
	'''Cuts off the existing model by the desired
	number of convolutions.
	There are 13 convolutions in total.
	We can choose any of the as the 'output'
	for our content model.
	'''
	model = VGG16_AvgPool(shape)
	my_model = Sequential()
	n = 0 # counter for conv layers
	for layer in model.layers:
		if layer.__class__ == Conv2D:
			n += 1
		my_model.add(layer)
		if n == num_convs:
			break

	print(my_model.summary())
	return my_model


def unpreprocess(img):
	'''VGG expets a different from RGB format for input.
	When we want to plot an image with Matplotlib, 
	it expects RGB format.
	This function basically does reverse of the 
	VGG preprocessing function.
	'''
	img[..., 0] += 103.939
	img[..., 1] += 116.779
	img[..., 2] += 126.68
	img = img[..., ::-1]
	return img


def scale_img(x):
	# for correct plotting with Matplotlib, rescale the intesity
	# of input image to the values in range [0, 1]
	x -= x.min()
	x = x/x.max()
	return x


if __name__ == '__main__':
	# load an image:
	# path = 'content/bale.jpeg'
	path = 'content/elephant.jpg'
	img = image.load_img(path)

	# convert image to an array:
	x = image.img_to_array(img)

	# preprocess the image for the vgg:
	x = np.expand_dims(x, axis=0) # 'cause image should be 4-D (1, H, W, C)
	x = preprocess_input(x)

	# take a look at the image:
	plt.subplot(1, 2, 1)
	plt.imshow(img)
	plt.title('original')

	plt.subplot(1, 2, 2)
	plt.imshow(x[0])
	plt.title('preprocessed')
	plt.show()

	# for later use:
	batch_shape = x.shape # since we have only one image
	shape = x.shape[1:] # to specify the shape of input to the model we need a 3-D format (H, W, C)
	
	# make a content model:
	# we will try different cutoffs of the vgg to see the resulting images:
	content_model = VGG16_AvgPool_CutOff(shape, 11)

	# making the target:
	# (recall, it's like theano's variables 
	# - a symbolic variable similar to thY we used before)
	target = K.variable(content_model.predict(x))


	# try to optimize wrt input:
	# (like in theano or tensorflow)
	# just the mean squared error between the symbolic content model output 
	# and the target we created above:
	loss = K.mean(K.square(target - content_model.output))

	# gradients wrt INPUT; needed for the optimizer:
	grads = K.gradients(loss, content_model.input) # also symbolic

	# similarly to theano.function:
	get_loss_and_grads = K.function(
		inputs=[content_model.input],
		outputs=[loss] + grads # because loss is expected to be a scalar,
	)                          # and grads is of the same size as input - 4-D


	def get_loss_and_grads_wrapper(x_vec):
		'''Scipy's minimizer allows us to pass back
		function value f(x) (loss in our case)
		and its gradient f'(x) simultaneously, rather then using
		the fprime argument (of the scipy function).

		We cannot use gel_loss_and_grads() directly, because
		input to the minimizer func must be a 1-D array.
		Input to get_loss_and_grads must be [batch_of_images].

		Gradient must also be a 1-D array, 
		and both loss and gradient must be np.float64.
		Otherwise the minimizer will give an error.
		'''
		l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
		return l.astype(np.float64), g.flatten().astype(np.float64)


	t0 = datetime.now()
	losses = []

	# randomly initialize the input to be optimized wrt to
	new_x = np.random.randn(np.prod(batch_shape))

	# training loop:
	for i in range(10):
		# return a new vector for x, the loss itself,
		# and another thing we don't care about:
		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
		new_x, l, _ = fmin_l_bfgs_b(
			func=get_loss_and_grads_wrapper,
			x0=new_x,
			maxfun=20 # this is going to let the f_min function do a few steps on its own
		)
		# remember we are going to alternate between minimizing and clipping:
		print('\niteration ', i, end=', ')
		# print('before clipping: new_x.min()=%f, new_x.max()=%f'%(new_x.min(), new_x.max()))		
		new_x = np.clip(new_x, -127, 127)
		# print('after clipping: new_x.min()=%f, new_x.max()=%f'%(new_x.min(), new_x.max()))
		losses.append(l)
		print('loss:', l)

	print('elapsed time:', datetime.now() - t0)

	# plot the losses:
	plt.plot(losses)	
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.show()

	# reshape the output and reverse the preprocessing for it:
	generated_img = new_x.reshape(*batch_shape)
	generated_img = unpreprocess(generated_img)

	# target_img = content_model.predict(x)
	# print('target_img.shape:', target_img.shape)
	

	# plot the original and generated images:
	plt.subplot(1, 2, 1)
	plt.imshow(img)
	plt.title('original')
	# scale the image for correct representation:
	plt.subplot(1, 2, 2)
	plt.imshow(scale_img(generated_img[0]))
	plt.title('reproduced')
	plt.show()	


