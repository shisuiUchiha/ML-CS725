from numpy import *

def gradient_descent(weights,i_train,o_train):
	grad=[0]*14
	lamda=0.2
	#print(weights[12])
	for j in range(0,14):
		ou_sum=0
		for i in range(0,400):
			summ=0
			for k in range(0,14):
				summ=summ+(weights[k]*i_train[i][k])
			#print(summ)
			i_summ=(o_train[i])-summ
			#print(summ)
			ou_sum += i_summ*(i_train[i][j])
			# print(ou_sum)
		grad[j]=2*lamda*weights[j] - 2*(ou_sum)
	# print(grad)
	return grad

def gradient_descent_p(weights,i_train,o_train,p):
	grad=[0]*14
	lamda=0.1
	p_value=p
	#print(weights[12])
	for j in range(0,14):
		ou_sum=0
		for i in range(0,400):
			summ=0
			for k in range(0,14):
				summ=summ+(weights[k]*i_train[i][k])
			#print(summ)
			i_summ=(o_train[i])-summ
			#print(summ)
			ou_sum += i_summ*(i_train[i][j])
			# print(ou_sum)
		#print(weights[j])
		if(weights[j]<0):
			k=-1
		else:
			k=1
		val=(abs(weights[j]))**(p_value-1.0)
		grad[j]=(k*p_value*lamda*val) - 2*(ou_sum)
	# print(grad)
	return grad


train_data=genfromtxt('train.csv',delimiter=',',skip_header=1)
i_test=genfromtxt('test.csv',delimiter=',',skip_header=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
#print (train_data.shape)
#print (train_data[399][0])

o_train=train_data[:,14]
o_train=array(o_train)
#print(o_train[0])
i_train=train_data[:,1:14]
stds = array([std(i_train[:,i]) for i in range(0,13)])
means  = array([mean(i_train[:,i]) for i in range(0,13)])
i_train = array([(i_train[:,i] - means[i])/stds[i] for i in range(0,13)])
i_test = array([(i_test[:,i] - means[i])/stds[i] for i in range(0,13)])
# o_train=array([o_train]).T

#print(o_train.shape)
b=ones((105,1))
i_test=i_test.T
i_test=concatenate((i_test,b),axis=1)


weights=random.normal(0,0.5,14)
weights[13]=1
weights=array(weights)

weights_p_1=random.normal(0,0.5,14)
weights_p_1[13]=1
weights_p_1=array(weights_p_1)

weights_p_2=random.normal(0,0.5,14)
weights_p_2[13]=1
weights_p_2=array(weights_p_2)


weights_p_3=random.normal(0,0.5,14)
weights_p_3[13]=1
weights_p_3=array(weights_p_3)



a=ones((400,1))
i_train=i_train.T
i_train=concatenate((i_train,a),axis=1)
#i_train=i_train.T

#print(i_train[1][1])
#print(i_train.shape)
#print(o_train.shape)


learning_rate=0.0002


for i in range(0,5000):
	#grad_o=(((weights).dot(i_train)) - o_train).dot(i_train.T)
	grad_o=gradient_descent(weights,i_train,o_train)
	# print(weights)
	#print(grad_o)	
	for j in range(0,14):
		weights[j]=weights[j]-learning_rate*grad_o[j]
#weights[0] -= sum(means/std)
#weights[1:] = weights[1:]/std
#print(weights)





#for p=1.2
for i in range(0,5000):
	#grad_o=(((weights).dot(i_train)) - o_train).dot(i_train.T)
	grad_o=gradient_descent_p(weights_p_1,i_train,o_train,1.2)
	# print(weights)
	#print(grad_o)	
	for j in range(0,14):
		weights_p_1[j]=weights_p_1[j]-learning_rate*grad_o[j]

#print(weights_p_1)


#for p=1.5
for i in range(0,5000):
	#grad_o=(((weights).dot(i_train)) - o_train).dot(i_train.T)
	grad_o=gradient_descent_p(weights_p_2,i_train,o_train,1.5)
	# print(weights)
	#print(grad_o)	
	for j in range(0,14):
		weights_p_2[j]=weights_p_2[j]-learning_rate*grad_o[j]

#print(weights_p_2)


#for p=1.8
for i in range(0,5000):
	#grad_o=(((weights).dot(i_train)) - o_train).dot(i_train.T)
	grad_o=gradient_descent_p(weights_p_3,i_train,o_train,1.8)
	# print(weights)
	#print(grad_o)	
	for j in range(0,14):
		weights_p_3[j]=weights_p_3[j]-learning_rate*grad_o[j]

#print(weights_p_3)

#for closed form

weights_c=linalg.lstsq(i_train,o_train)[0]
#print(weights_c)


output=dot(i_test,weights.T)
output_p_1=dot(i_test,weights_p_1.T)
output_p_2=dot(i_test,weights_p_2.T)
output_p_3=dot(i_test,weights_p_3.T)
output_c=dot(i_test,weights_c.T)


#print(output.shape)
#print(output_p_1.shape)
#print(output_p_2.shape)
#print(output_p_3.shape)



output=array(output,ndmin=2)
output=output.T
#print(output.shape)


output_p_1=array(output_p_1,ndmin=2)
output_p_1=output_p_1.T
#print(output_p_1.shape)


output_p_2=array(output_p_2,ndmin=2)
output_p_2=output_p_2.T
#print(output_p_2.shape)


output_p_3=array(output_p_3,ndmin=2)
output_p_3=output_p_3.T
#print(output_p_3.shape)

output_c=array(output_c,ndmin=2)
output_c=output_c.T


rows=output.shape[0]
#output.item

#print(rows)





a=[]
for i in range(0,rows):
	a.append(i)

a=asmatrix(a)
a=a.transpose()

final=hstack((a,output))
final_p_1=hstack((a,output_p_1))
final_p_2=hstack((a,output_p_2))
final_p_3=hstack((a,output_p_3))
final_c=hstack((a,output_c))

#savetxt("foo3.csv",(a,output),fmt='%i,%f',comments='',header="ID,MEDV", delimiter=",")

savetxt('output.csv',final,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')
savetxt('output_p1.csv',final_p_1,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')
savetxt('output_p2.csv',final_p_2,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')
savetxt('output_p3.csv',final_p_3,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')
savetxt('output_closed.csv',final_c,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')