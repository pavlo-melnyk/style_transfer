from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from skimage.transform import resize

import keras.backend as K 
import numpy as np 
import matplotlib.pyplot as plt

from reproducing_content import VGG16_AvgPool, VGG16_AvgPool_CutOff, unpreprocess, scale_img
from recreating_style import gram_matrix, style_loss, minimize
from scipy.optimize import fmin_l_bfgs_b



def load_and_preprocess_img(path, shape=None):
	'''Loads and prepocesses an image.
	'''
	img = image.load_img(path, target_size=shape)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x


if __name__ == '__main__':
	# load and preprocess a content image:
	content_img = load_and_preprocess_img(
		# 'content/lena.png',
		# 'content/elephant.jpg',
		# 'content/batman.jpg',
		# 'content/islands.jpg',
		'content/beautiful-sceneries-sunrise.jpg',
		# 'content/bale.jpeg',
		# 'content/sydney.jpg'
	)

	# plt.imshow(scale_img(unpreprocess(content_img)[0]))
	# plt.title('content image')
	# plt.show()

	# grap the height and width of the content image
	# in order to  resize the style image, 
	# since we need it to be of the same size with the content_img
	# (to calculate the MSE), and don't care much about warping it:
	h, w = content_img.shape[1:3]
	# we may leave it as-is, because the gram matrix (before the MSE calculation)
	# is of CxC size
	
	# load and preprocess a style image:
	style_img = load_and_preprocess_img(
		# 'styles/wheatfield_with_crows.jpg',
		'styles/starrynight.jpg',
		# 'styles/lesdemoisellesdavignon.jpg',
		# 'styles/picasso.jpg',
		# 'styles/Whaet-Field-with-Cypresses.jpg',
		# 'styles/monalisa.jpg',
		# 'styles/courbet.jpg',
		(h, w)

	)

	# plt.imshow(scale_img(unpreprocess(style_img)[0]))
	# plt.title('style image')
	# plt.show()

	# print(style_img.shape)

	# for later use:
	batch_shape = content_img.shape
	shape = content_img.shape[1:]
	# we need only one VGG here:
	vgg = VGG16_AvgPool(shape)

	# create the content model;
	# we need only 1 output:
	# print(vgg.summary())
	content_model = Model(vgg.input, vgg.layers[13].get_output_at(1))
	print(content_model.summary())
	content_model_target = K.variable(content_model.predict(content_img))

	# create the style model along with the content one;
	# recall, we want multiple outputs:
	symbolic_outputs = [
		layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')
	]
	# assert len(symbolic_outputs) == 5

	# create the style_model with myltiple outputs:
	style_model = Model(vgg.input, symbolic_outputs)

	# calculate the targets of the style_model:
	style_layers_targets = [K.variable(t) for t in style_model.predict(style_img)]

	# assuming, that the weight of the content loss is 1,
	# we are going to weight only the style losses:
	# style_losses_weights = [0.2, 0.4, 0.3, 0.5, 0.2]
	style_losses_weights = [1, 1, 1, 1, 1]

	# create the total loss which is the sum of content + style loss
	loss = K.mean(K.square(content_model_target - content_model.output))

	for w, symbolic, actual in zip(style_losses_weights, symbolic_outputs, style_layers_targets):
		# recall, gram_matrix() expects a (H, W, C) as input:
		loss += w * style_loss(symbolic[0], actual[0])

	# again, we create the gradient_and_loss_function;
	# NOTE: no matter which model's input to use - 
	#       they are both pointing to the same keras
	#       Input layer in memory
	grads = K.gradients(loss, vgg.input) # a list of grads

	# like Theano function:
	get_loss_and_grads = K.function(
		inputs=[vgg.input],
		outputs=[loss] + grads
	)

	# as in reproducing_content.py and recreating_style.py
	def get_loss_and_grads_wrapper(x_vec):
		l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
		return l.astype(np.float64), g.flatten().astype(np.float64)

	generated_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)

	plt.imshow(scale_img(generated_img))
	plt.title('generated image')
	plt.show()

	plt.subplot(1, 3, 1)
	plt.imshow(scale_img(unpreprocess(content_img)[0]))
	plt.title('content image')
	

	plt.subplot(1, 3, 2)
	plt.imshow(scale_img(unpreprocess(style_img)[0]))
	plt.title('style image')
	
	plt.subplot(1, 3, 3)
	plt.imshow(scale_img(generated_img))
	plt.title('generated image')
	plt.show()

	plt.imsave('generated_img3.jpg', scale_img(generated_img))




