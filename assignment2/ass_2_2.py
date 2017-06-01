import pandas as pd
import numpy as np

seed = 7
np.random.seed(seed)
in_data=pd.read_csv("train.csv")
pred_data=pd.read_csv("kaggle_test_data.csv")

#print in_data.get_value(1,1)

out_data=pd.get_dummies(in_data)   #nominal values encoded to numerical
pred_out=pd.get_dummies(pred_data)
out_csv=pd.DataFrame(out_data)
pred_csv=pd.DataFrame(pred_out)
j=0
missing_index=[];
for i in range(110):
	if pred_csv.keys()[j]==out_csv.keys()[i]:
		j=j+1
	elif out_csv.keys()[i]=="salary":
		j=j
	else:
		print out_csv.keys()[i]
		missing_index.append(i)
print missing_index
out_csv.to_csv('a.csv')
pred_csv.to_csv('b.csv')
out_data=out_data.values               #converting pandas dataframe to numpyarray
pred_out=pred_out.values
#print out_data[2,:]
for i in missing_index:
	print i
	pred_out=np.insert(pred_out,i-1,0,axis=1)
outdata=out_data
y=outdata[:,7]
outdata=np.delete(outdata,7,1)
print outdata.shape
print pred_out.shape
out_data=outdata
print pred_out[2,:]
print outdata[2,:]
pred_mean=np.mean(pred_out,axis=0)
pred_std=np.std(pred_out,axis=0)
mean1=np.mean(out_data,axis=0)
std1=np.std(out_data,axis=0)
pred_out=pred_out.astype(float)
outdata=outdata.astype(float)
#print outdata[:,1]
'''
outdata[:,1]=(outdata[:,1]-mean[1])/std[1]
print outdata[:,1]
print outdata[:,2]
outdata[:,2]=(outdata[:,2]-mean[2])/std[2]
print outdata[:,2]
print outdata[:,3]
outdata[:,3]=(outdata[:,3]-mean[3])/std[3]
print outdata[:,3]
print outdata[:,4]
outdata[:,4]=(outdata[:,4]-mean[4])/std[4]
print outdata[:,4]
print outdata[:,5]
outdata[:,5]=(outdata[:,5]-mean[5])/std[5]
print outdata[:,5]
print outdata[:,6]
outdata[:,6]=(outdata[:,6]-mean[6])/std[6]
print outdata[:,6]
'''
for i in range(1,7):
	#print outdata[:,i]
	pred_out[:,i]=(pred_out[:,i]-pred_mean[i])/pred_std[i]
	outdata[:,i]=(outdata[:,i]-mean1[i])/std1[i]
	#print outdata[:,i]
outdata=np.delete(outdata,0,1)
id_pred=pred_out[:,0]
pred_out=np.delete(pred_out,0,1)
#print outdata
#print y
np.savetxt("foo2.csv",pred_out,delimiter=",")
np.savetxt("foo.csv",outdata, delimiter=",")

dimn=pred_out.shape[1]
size=outdata.shape[0]
test_size=pred_out.shape[0]


#preprocessing and all the shit has been completed by here 
#main things are outdata and pred_out thats all.

#lets write for forward pass
batch_size=20
layer_size=54
w1= np.random.standard_normal((dimn, layer_size))
w2= np.random.standard_normal((layer_size,1))
fin_error= np.zeros(shape=(layer_size,1))
dummy1=np.random.standard_normal((layer_size,1))
dummy2=np.random.standard_normal((dimn,layer_size))
fin2_error=np.zeros(shape=(dimn,layer_size))
fin_error=np.asmatrix(fin_error)
fin2_error=np.asmatrix(fin2_error)
w1=np.asmatrix(w1)
w2=np.asmatrix(w2)
fin_error=np.asmatrix(fin_error)
fin2_error=np.asmatrix(fin2_error)
dummy1=np.asmatrix(dummy1)
dummy2=np.asmatrix(dummy2)
#bias1=(np.zeros(shape=(1,layer_size))/100000)*(-1)
#bias1=np.asmatrix(bias1)
#bias2=(np.zeros(shape=(1,1))/100000)*(-1)
#bias2=np.asmatrix(bias2)
#print w2.shape

def activation_1(data):
	#print data
	data=np.exp(-1*data)
	e_data=data
	#print data.item((0,0))
	for i in range(data.shape[1]):
		data[0,i]=(1/(1+data[0,i]))
		e_data[0,i]=(e_data[0,i])*(data[0,i])*(data[0,i])
	#print data
	return e_data,data


def activation_2(data):
	#print data
	data=np.exp(-1*data)
	#print data
	e_data=data
	#print data.item((0,0))
	#print data.shape
	for i in range(data.shape[1]):
		data[0,i]=(1/(1+data[0,i]))
		#print "heyy"
		#print data
		#print e_data
		e_data[0,i]=(e_data[0,i])*(data[0,i])*(data[0,i])
	#print data
	return e_data,data


def forward_pass(data,w_1,w_2,index,fin_error,fin2_error):
	l1=data.dot(w_1)
	#print l1.shape 
	#l1=l1+bias1
	l1_e_data,l1_a=activation_1(l1)
	l2=l1_a.dot(w_2)
	#l2=l2+bias2
	l2_e_data,l2_a=activation_2(l2)
	error=y[index]-l2_a
	#print "olaola"
	for j in range(fin2_error.shape[0]):
		for k in range(fin2_error.shape[1]):
			fin2_error[j,k]=-1*2*(error)*l2_e_data[0,0]*w_2[k,0]*data[0,j]*l1_e_data[0,k]
	for i in range(fin_error.shape[1]):
		fin_error[0,i]=-1*2*(error)*(l2_e_data)*l1_a[0,i]
	return fin_error,fin2_error



def backpropagation(w_1,w_2,fin_error,fin2_error,learning_rate):
	for i in range(w_2.shape[0]):
		w_2[i,0]=w_2[i,0]-(learning_rate*fin_error[i,0])
	for j in range(w_1.shape[0]):
		for k in range(w_1.shape[1]):
			w_1[j,k]=w_1[j,k]-(learning_rate*fin2_error[j,k])
	return w_1,w_2



j=0
error=0
for i in range(size):
	if j<batch_size:
		a=outdata[i,:]
		a=np.asmatrix(a)
		dummy1,dummy2=forward_pass(a,w1,w2,i,fin_error,fin2_error)
		fin2_error=fin2_error+dummy2
		j=j+1
	else:
		j=0
		w1,w2=backpropagation(w1,w2,fin_error,fin2_error,0.2)
		error=0
		for j in range(w1.shape[0]):
			for k in range(w1.shape[1]):
				fin2_error[j,k]=0
		for i in range(w2.shape[0]):
			fin_error[i,0]=0
print "completed"
print w2
print w1


def activation_test(data):
	data=np.exp(-1*data)
	for k in range(data.shape[1]):
		data[0,i]=(1/(1+data[0,i]))
	return data


y=[]
for k in range(test_size):
	a=pred_out[k,:]
	a=np.asmatrix(a)
	fl=a.dot(w1)
	#fl=fl+bias1
	fl=activation_test(fl)
	ol=fl.dot(w2)
	x=ol[0,0]
	#x=x+bias2[0,0]
	x=(1/(1+np.exp(-1*x)))
	print x
	y.append(x)


for i in range(len(y)):
	if(y[i]<0.5):
		y[i]=0
	else:
		y[i]=1

id_pred=np.matrix(id_pred)
y=np.asmatrix(y)
output=np.concatenate((id_pred.T,y.T),axis=1)
np.savetxt("final_without_lib.csv",output,header="id,salary",delimiter=",",comments='',fmt='%i')



