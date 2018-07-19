import numpy as np 
np.random.seed(1996)

from keras.models import Model, Sequential
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16 

from reproducing_content import VGG16_AvgPool, unpreprocess, scale_img
from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime

import matplotlib.pyplot as plt 
import keras.backend as K 


def gram_matrix(img):
	'''Takes as input 3-D image (H, W, C).
	Returns (CxC) Gram Matrix matrix.
	'''
	# convert the input image to (C, H*W)
	X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))

	# Gram Matrix calculation:
	# gram = X.dot(X.T) / N
	G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
	# tge division is not necessary, since we are going to calculate
	# the mean square error anyway
	return G


def style_loss(t, y):
	return K.mean(K.square(gram_matrix(t) - gram_matrix(y)))


# similar to reproducing_contetn's training loop;
# has been put to a function to make it reusable later on:
def minimize(fn, epoch, batch_shape):
	t0 = datetime.now()
	losses = []
	x = np.random.randn(np.prod(batch_shape))
	for i in range(epoch):
		x, l, _ = fmin_l_bfgs_b(
			func=fn,
			x0=x,
			maxfun=20
		)

		print('\nepoch: %d,  loss: %.6f' % (i, l))
		# [print('loss at conv%s_1:'%i, ll) for i, ll in zip(range(1,6), get_each_layer_losses([x.reshape(*batch_shape)]))]
		# print('loss check:', get_loss_and_grads([x.reshape(*batch_shape)])[0])
		x = np.clip(x, -127, 127)		

		losses.append(l)

	print('duration:', datetime.now() - t0)
	plt.plot(losses)
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.show()

	generated_img = x.reshape(*batch_shape)
	generated_img = unpreprocess(generated_img)
	return generated_img[0]


if __name__ == '__main__':
	# path = 'styles/lesdemoisellesdavignon.jpg'
	path = 'styles/picasso.jpg'
	
	# load the image:
	img = image.load_img(path)

	# convert the image to array:
	x = image.img_to_array(img)

	# expand dimensions to make 4-D (1, H, W, C):
	x = np.expand_dims(x, axis=0)

	# preprocess for VGG:
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
	batch_shape = x.shape 
	shape = batch_shape[1:]

	# we want to use the whole VGG16_AvgPool from the reproduce_content.py;
	# we'll take the first convolution of every conv block for our target output:
	vgg = VGG16_AvgPool(shape)

	# recall, for reproducing style, we need to use multiple outputs of the vgg:
	symbolic_conv_outputs = [
		layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')
	]
	'''Why layer.get_output_at(1) instead of layer.output?
	When we made our VGG16_AvgPool earlier, we replaced all the MaxPools
	with AvgPools. So we created a new model using layers of an existing model.
	So in memory, Keras sees that there are actually two existing models and 
	they share some layers. 
	So the layers have two different outputs because there are two different models
	representing two different paths through the layers, depending on whether we're
	using the MaxPool model or the AvgPool model.
	Since the MaxPool model was created first, it gets index 0 (layer.get_output_at(1)), 
	and the AvgPool model gets index 1 (layer.get_output_at(1)).
	'''

	# we could select a subset of the outputs to see what kind of effect it has:
	# symbolic_conv_outputs = symbolic_conv_outputs[:2]
	'''Remember, that a CNN's earlier layers find smaller, 'localized', features,
	whereas the later layers find larger, more 'global', features.
	'''

	# create the model we're to use:
	multi_output_model = Model(vgg.input, symbolic_conv_outputs)
	# print(multi_output_model.summary())

	# let's calculate the targets, which are outputs at each layer:
	style_layers_outputs = [K.variable(y) for y in multi_output_model.predict(x)]
	# multi_output_model.predict(x) - is a np.array of the actual values

	# let's weight each output's loss:
	losses_weights = [1, 1, 1, 1, 1]

	# calculate the style loss:
	loss = 0
	ls = []
	i = 0
	for w, symbolic, actual in zip(losses_weights, symbolic_conv_outputs, style_layers_outputs):
		l = w * style_loss(symbolic[0], actual[0])		
		loss += l
		ls.append(l)
		
		# because gram_matrix() function expects a 3-D (H, W, C) as input

	# take the gradient with respect to the input:
	grads = K.gradients(loss, multi_output_model.input) # a list of gradients

	# similar to theano.function:
	get_loss_and_grads = K.function(
		inputs=[multi_output_model.input],
		outputs=[loss] + grads
	)

	get_each_layer_losses = K.function(
		inputs=[multi_output_model.input],
		outputs=ls
	)

	def get_loss_and_grads_wrapper(x_vec):
		l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
		return l.astype(np.float64), g.flatten().astype(np.float64)

	generated_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
	plt.subplot(1, 2, 1)
	plt.imshow(img)
	plt.title('original')

	plt.subplot(1, 2, 2)
	plt.imshow(scale_img(generated_img))
	plt.title('recreated style')
	plt.show()