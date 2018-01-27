# coding: utf-8
import tensorflow as tf
import librosa
import numpy as np
from sys import stderr
from loader import read_audio_spectrum

class neural_audio_style_transfer(object):

	def __init__(self, content, style, n_fft = 2048, n_filter = 4096):

		self.n_filter = n_filter
		self.n_fft = n_fft

		self.a_content, self.sr = read_audio_spectrum(content)
		print('Content {0} is loaded successfully!'.format(content))
		self.a_style, self.sr = read_audio_spectrum(style)

		self.n_channels, self.n_samples = self.a_content.shape
		self.a_style = self.a_style[:self.n_channels, :self.n_samples]
		print('Style {0} is loaded successfully!'.format(style))

		self.a_content_tf = np.ascontiguousarray(self.a_content.T[None, None, :,:])
		self.a_style_tf = np.ascontiguousarray(self.a_style.T[None, None,:,:])

		self.prng = np.random.RandomState(0)

		std = np.sqrt(2) * np.sqrt(2/(self.n_channels + self.n_filter) * 11)
		self.kernel = self.prng.randn(1, 11, self.n_channels, self.n_filter) * std


	def feats(self):

		g = tf.Graph()
		with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:

			x = tf.placeholder('float32', [1,1,self.n_samples, self.n_channels], name='x')
			kernel_tf = tf.constant(self.kernel, name = "kernel", dtype='float32')

			conv = tf.nn.conv2d(x, kernel_tf, strides=[1,1,1,1], padding='VALID', name='conv')

			net = tf.nn.relu(conv)

			content_features = net.eval(feed_dict = {x:self.a_content_tf})
			style_features = net.eval(feed_dict = {x:self.a_style_tf})

			features = np.reshape(style_features, (-1, self.n_filter))
			style_gram = np.matmul(features.T, features) / self.n_samples

			return content_features, style_gram




	def optimize(self, alpha=0.01, learning_rate=0.001):
		
		self.alpha = alpha
		self.lr = learning_rate

		content_features, style_gram = self.feats()
		self.result = None

		with tf.Graph().as_default():

			x = tf.Variable(self.prng.randn(1,1,self.n_samples,self.n_channels).astype(np.float32)*1e-3, name ='x')

			kernel_tf = tf.constant(self.kernel, name='kernel', dtype='float32')
			conv = tf.nn.conv2d(x, kernel_tf, strides=[1,1,1,1], padding='VALID',name='conv')

			net = tf.nn.relu(conv)

			content_loss = self.alpha * 2 * tf.nn.l2_loss(net - content_features)

			style_loss = 0

			_, height, width, number = map(lambda i: i.value, net.get_shape())

			size = height * width * number
			features = tf.reshape(net, (-1, number))
			gram = tf.matmul(tf.transpose(features), features)/ self.n_samples
			style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

			loss = content_loss + style_loss
			opt = tf.contrib.opt.ScipyOptimizerInterface(loss, method = 'L-BFGS-B', options={'maxiter':300})

			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())

				print('Optimization starts!')
				opt.minimize(sess)

				print('Final loss:', loss.eval())

				self.result = x.eval()

			
	def save(self, output = 'outputs/out.wav'):


		a = np.zeros_like(self.a_content)
		a[:self.n_channels,:] = np.exp(self.result[0,0].T) - 1
		p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi

		self.a_result = a
	
		for i in range(500):
			S = a * np.exp(1j*p)
			x = librosa.istft(S)
			p = np.angle(librosa.stft(x, self.n_fft))

		librosa.output.write_wav(output, x, self.sr)
		print('File is saved successfully!')


































