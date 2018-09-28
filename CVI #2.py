import tensorflow as tf
import numpy as np;
import datetime as dt
import cv2
import glob
import sys
n_inputs = 7990  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
	#print(
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 50
batch_size = 50
total=0
inp=[]
outp=[]
with tf.Session() as sess:
	init.run()
	for num in range(10):
		count=0;
		for i in glob.glob('E:\dataset\TRAINING DATA/NUMBERS/'+str(num)+'\*'):  #you can give any path here

			if count<1000:
				h=cv2.imread(i)
				b,g,r=cv2.split(h)
				input_vector=np.array(r).reshape(7990)
				for g in range(len(input_vector)):
					if input_vector[g]==255:
						input_vector[g]=1
				inp.append(input_vector)
				output_vector=num
				outp.append(output_vector)
				total=total+1
			count=count+1
	for epoch in range(n_epochs):
		for iteration in range(total // batch_size):
			if(iteration!=0):
				X_batch=inp[iteration:iteration*batch_size]
			else:
				X_batch=inp[iteration:batch_size]
				
			if(iteration!=0):
				y_batch=outp[iteration:iteration*batch_size]
			else:
				y_batch=outp[iteration:batch_size]
				
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})        
		print(epoch, "Train accuracy:", acc_train)
		#save_path = saver.save(sess, "E:\Projects\ML\TRAffic\my_model_final.ckpt")
	while 1:
			folder=input("Enter folder name...")
			img=input("Enter image name...")
			h=cv2.imread('E:\dataset\TRAINING DATA/NUMBERS/'+folder+'/'+img+'.png')  #you can give any path here
			b,g,r=cv2.split(h)
			input_vector=np.array(r).reshape(1,7990)
			for g in range(len(input_vector)):
					if input_vector[0,g]==255:
						input_vector[0,g]=1
			
			X_new_scaled = input_vector  # some new images (scaled from 0 to 1)
			Z = logits.eval(feed_dict={X: X_new_scaled})
			y_pred = np.argmax(Z, axis=1)
			print(y_pred)
	
	
  



