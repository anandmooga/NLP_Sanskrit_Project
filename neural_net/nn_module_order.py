import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import numpy as np
import time,os
class NN_Order:
	def __init__(self,input_layer_dim,hidden_layer_dim, output_layer_dim, lambda1,lambda2,keep_prob):
		tf.set_random_seed(123)
		### Setting up the dimensions here
		self.input_layer_dim = input_layer_dim
		self.hidden_layer_dim = hidden_layer_dim
		self.output_layer_dim = output_layer_dim
		self.lambda1 = lambda1
		self.lambda2 = lambda2
		self.keep_prob = keep_prob

		### Neural Network Architecture Below
		self.x  = tf.placeholder(tf.float32, (None,input_layer_dim))
		self.z1 = tf.layers.dense(inputs = self.x,units = hidden_layer_dim, activation = tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self.lambda2))
		self.z1 = tf.nn.dropout(self.z1,keep_prob = keep_prob)
		self.y = tf.layers.dense(self.z1,units = output_layer_dim)
		
		### Loss definition below
		self.saver = tf.train.Saver()
		self.flags = tf.placeholder(tf.float32, (None, self.output_layer_dim))
		self.loss = tf.multiply(self.y, self.flags)
		self.train_step = tf.train.GradientDescentOptimizer(self.lambda1).minimize(self.loss)
		# self.label = tf.placeholder(tf.float32, (None,self.output_layer_dim))
		# self.cross_entropy = -tf.reduce_mean(self.label * tf.log(self.y), axis=1)	
		# self.ypred = tf.placeholder(tf.float32, shape=[None, self.output_layer_dim])
		# self.ytarg = tf.placeholder(tf.float32, shape=[None, self.output_layer_dim])
		# self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.ytarg,1))
		# self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		# self.loss = (self.cross_entropy) 

	def forward(self,X):
		with tf.Session() as sess:
			self.saver.restore(sess, "weights/model.ckpt")
			Y = self.y.eval(feed_dict={
					self.x: np.reshape(X,(1,784))
				})
			print(Y)
			return tf.argmax(Y,axis=1)

	# def backprop(self,flag,X):
		## flag is either positive or negative depending on whether its a gt edge or a wrongly predicted edge
		# self.train_step = tf.train.GradientDescentOptimizer(self.lambda1).minimize(flag*self.Y)
		# with tf.Session() as sess:
		# 	sess.run(self.train_step,feed_dict={
		# 			self.x = X
		# 		})

		

	def train(self,train_images,train_labels_onehot,val_images,val_labels_onehot,num_epochs,batch_size):
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			max_acc = 0
			max_acc_epoch = -1
			print("Started Training")
			for epoch in range(num_epochs):
				st = time.time()
				print("Beginning Epoch : {}/{}".format(epoch,num_epochs))
				for i in range(int(50000/batch_size)):
					sess.run(self.train_step, feed_dict={
						self.x:train_images[i*batch_size:(i+1)*batch_size], 
						self.label: train_labels_onehot[i*batch_size:(i+1)*batch_size] 
					})
					if(i%100==0):
						cur_acc = self.accuracy.eval(feed_dict = {
							self.x: val_images[:],
							self.ytarg: val_labels_onehot
						})
						print("Epoch: {}/{}, Iter: {}/{} Validation Accuracy: {}".format(epoch+1,num_epochs,i,int(50000/batch_size),cur_acc))
						
				cur_acc = self.accuracy.eval(feed_dict = {
							self.x: val_images[:],
							self.ytarg: val_labels_onehot
						})
				print(time.time()-st,"seconds")
				save_path = self.saver.save(sess, "weights/model.ckpt")
				if(cur_acc>max_acc):	
					max_acc = cur_acc
					max_acc_epoch = 0
				else:
					if(epoch > max_acc_epoch+patience_epoch):
						print("Applying Early Stopping...")
						break

	def test(self,test_images,test_labels_onehot):
		with tf.Session() as sess:
			self.saver.restore(sess, "weights/model.ckpt")
			fin_acc = self.accuracy.eval(feed_dict = {
				self.x: test_images[:],
				self.ytarg: test_labels_onehot
			})
			return fin_acc