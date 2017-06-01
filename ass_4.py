from numpy import *





def formatting_train(data):
	input_matrix=[]
	output_matrix=[]
	i=0
	for line in data:
		words=line.split(',')
		if (i!=0):
			h=len(words)-1
			input_list=words[1:h]
			input_list=list(map(float,input_list))
			#print (type(input_list))
			string=words[h]
			string=string[0:len(string)-1]
			string=float(string)
			output_matrix.append(string)
			input_matrix.append(input_list)
		i=i+1
	input_matrix=asmatrix(input_matrix)
	output_matrix=asmatrix(output_matrix)
	return input_matrix,output_matrix.transpose()

def formatting_test(data):
	input_matrix=[]
	i=0
	for line in data:
		words=line.split(',')
		if (i!=0):
			j=len(words)
			input_list=words[1:j]
			h=len(input_list)
			string=input_list[h-1]
			string=string[0:len(string)-1]
			string=float(string)
			input_list[h-1]=string
			input_list=list(map(float,input_list))
			#print (type(input_list))
			input_matrix.append(input_list)
		i=i+1
	input_matrix=asmatrix(input_matrix)
	return input_matrix




def gradient_descent(weights,i_train,o_train):
	grad=[0]*13
	lamda=0
	for j in range(0,13):
		ou_sum=0
		for i in range(0,400):
			summ=0
			for k in range(0,13):
				summ=summ+(weights[k]*i_train.item((i,k)))
			#print(summ)
			i_summ=o_train.item((i,0))-summ
			#print(summ)
			ou_sum=ou_sum+(i_train.item((i,j))*i_summ)
			#print(ou_sum)
		grad[j]=-2*(ou_sum)+2*lamda*weights[j]
	#print(grad)
	return grad



train_data=open("train.csv","r")
i_train,o_train=formatting_train(train_data)

#print (i_train)
#print (o_train)

x=linalg.lstsq(i_train,o_train)[0]

test_data=open("test.csv","r")
i_test=formatting_test(test_data)

print (i_train.shape)
print (o_train.shape)


weights=[0]*13
weights=array([weights]).T
learning_rate=0.002


for i in range(0,100):
	grad_o=gradient_descent(weights,i_train,o_train)
	for j in range(0,13):
		weights[j]=weights[j]-learning_rate*grad_o[j]


weights=asmatrix(weights)
weights=weights.transpose()

print(weights)

output=dot(i_test,weights.T)

print(output)

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

savetxt('b.csv',final,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')


