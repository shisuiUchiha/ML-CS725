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


train_data=open("train.csv","r")
i_train,o_train=formatting_train(train_data)

#print (i_train)
#print (o_train)

x=linalg.lstsq(i_train,o_train)[0]

test_data=open("test.csv","r")
i_test=formatting_test(test_data)

output=dot(i_test,x)



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

savetxt('a.csv',final,fmt='%i,%f',delimiter=",",header="ID,MEDV",comments='')




