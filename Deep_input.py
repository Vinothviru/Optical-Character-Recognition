import cv2
import glob
import copy as c
import Deep_NN as Deep
import numpy as np
print("******************************************CHARACTER RECOGNITION USING SINGLE LAYERED NEURAL NETWORKS******************************************")
test=Deep.Deep_NN(7990,5000,15,1,1)
iter=0
numo=0
i_list=np.zeros((7990,5000))
output_vector=np.zeros((1,5000))
for num in range(2):
	count=0
	for i in glob.glob('E:\dataset\TRAINING DATA/NUMBERS/'+str(num)+'\*'):
		if count<2500:
				#print(iter)
				img=cv2.imread(i)
				b,g,r=cv2.split(img)
				input_vector=np.array(r).reshape(7990,1)
				input_vector[input_vector==255]=1	
				if(numo==0):
					output_vector[:,:2500]=1
					numo+=1
				i_list[:,iter]=input_vector[:,0]
				#o_list[:,iter]=output_vector.reshape(1)	
		count=count+1	
		#iter+=1
	
#print(output_vector)		
learning_rate=0.1                                     
test.Gradient_descent(learning_rate,i_list,output_vector)
l=1
print("\n")
print("**HAI I AM TRAINED TO CLASSIFY THE CHARACTERS...BUT HERE YOU TRAINED ME TO CLASSIFY THE NUMBERS '0' AND '1'. TRAINED POSITIVELY FOR '0' AND NEGATIVELY FOR '1'.**\n")
print("\n")
print("******************TEST ME IF YOU DON'T TRUST ME...*********************")
print("\n")
while 1:
	samp=0
	print("SELECT THE FOLDER NAME WHICH HAS TO BE RECOGNIZED")
	val=int(input())
	print("\n")
	print("GIVE THE IMAGE FOR TESTING...SELECT AN IMAGE BETWEEN 1 AND 1000 WHICH ARE AVAILABLE IN MY DATA SET" )
	im=int(input())
	print("\n")
	#print("hello")
	for j in glob.glob('E:\dataset\TRAINING DATA/NUMBERS/'+ str(val) +'/'+str(im)+'.png'):
		img1=cv2.imread(j)
		b,g,r=cv2.split(img1)
		red_value1=np.array(r)
		samp_input_vector=red_value1.reshape(7990,1)
		samp=np.array((samp_input_vector))
		samp[samp==255]=1
	Test_output=test.Forward_propogation(samp)
	if(Test_output>0.5):
		print("PREDICTED VALUE : ",Test_output)
		print("YES THE PREDICTED VALUE IS CLOSE TO 1,I AM PREDICTING THE INPUT IMAGE WOULD BE 0")
	else:
		print("PREDICTED VALUE : ",Test_output)
		print("YES THE PREDICTED VALUE IS CLOSE TO 0,I AM PREDICTING THE INPUT IMAGE WOULD BE 1")
	print("\n")
	l+=1
	fin=int(input("DO YOU WANNA TEST ME AGAIN??? PRESS 1    "))
	print("\n")
	if(fin==1):
		continue
	else:
		break
print("*************************************************MY JOB IS DONE CHIEF********************************************")