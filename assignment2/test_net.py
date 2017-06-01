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


w1=np.zeros((108,50))
w1=np.asmatrix(w1)
w2=np.zeros((50,1))
w2=np.asmatrix(w2)

a=np.genfromtxt("weights.txt",delimiter=",")
a=np.asmatrix(a)

for i in range(w1.shape[0]):
	for j in range(w1.shape[1]):
		w1[i,j]=a[i,j]


k=w1.shape[0]
for i in range(w2.shape[0]):
	w2[i,0]=a[k+i,0]

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
np.savetxt("predictions.csv",output,header="id,salary",delimiter=",",comments='',fmt='%i')