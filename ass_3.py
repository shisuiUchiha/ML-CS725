from numpy import *

def gradient_descent(weights,i_train,o_train):
	grad=[0]*14
	lamda=5
	#print(weights[12])
	for j in range(0,14):
		ou_sum=0
		for i in range(0,201):
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


train_data=genfromtxt('small_train.csv',delimiter=',',skip_header=1)
i_test=genfromtxt('small_test.csv',delimiter=',',skip_header=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
#print (train_data.shape)
#print (train_data[399][0])

o_train=train_data[:,14]
o_train=array(o_train)
print(o_train[0])
i_train=train_data[:,1:14]
stds = array([std(i_train[:,i]) for i in range(0,13)])
means  = array([mean(i_train[:,i]) for i in range(0,13)])
i_train = array([(i_train[:,i] - means[i])/stds[i] for i in range(0,13)])
i_test = array([(i_test[:,i] - means[i])/stds[i] for i in range(0,13)])
# o_train=array([o_train]).T

#print(o_train.shape)
b=ones((198,1))
i_test=i_test.T
i_test=concatenate((i_test,b),axis=1)
weights=random.normal(0,0.5,14)
weights[13]=0.2
weights=array(weights)
a=ones((201,1))
i_train=i_train.T
i_train=concatenate((i_train,a),axis=1)
#i_train=i_train.T

print(i_train[1][1])
print(i_train.shape)
print(o_train.shape)


learning_rate=0.0003


for i in range(0,1000):
	#grad_o=(((weights).dot(i_train)) - o_train).dot(i_train.T)
	grad_o=gradient_descent(weights,i_train,o_train)
	# print(weights)
	#print(grad_o)	
	for j in range(0,14):
		weights[j]=weights[j]-learning_rate*grad_o[j]
#weights[0] -= sum(means/std)
#weights[1:] = weights[1:]/std
print(weights)

output=dot(i_test,weights.T)

print(output)
print(output.shape)

output=array(output,ndmin=2)
output=output.T
print(output.shape)


rows=output.shape[0]
#output.item

print(rows)



a=[]
for i in range(0,rows):
	a.append(i)

a=asmatrix(a)
a=a.transpose()

final=hstack((a,output))

#savetxt("foo3.csv",(a,output),fmt='%i,%f',comments='',header="ID,MEDV", delimiter=",")

savetxt('chinga.csv',final,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')