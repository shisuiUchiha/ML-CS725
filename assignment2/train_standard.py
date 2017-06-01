from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
import pandas as pd
import numpy as np


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

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(outdata,y)
predicted = knn.predict(pred_out)
print predicted

a=np.matrix(predicted)
id_pred=np.matrix(id_pred)
print a.shape
print type(a)
print id_pred.shape
print type(id_pred)

output=np.concatenate((id_pred.T,a.T),axis=1)

print output
print output.shape

output=np.asarray(output)

print output.shape
print type(output)


np.savetxt("predictions_1.csv",output,header="id,salary",delimiter=",",comments='',fmt='%i')



dimn=pred_out.shape[1]


#preprocessing and all the shit has been completed by here 
#main things are outdata and pred_out thats all.

seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Dense(dimn+4, input_dim=dimn, init='uniform', activation='relu'))
model.add(Dense(dimn, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

model.fit(outdata,y, nb_epoch=100, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(pred_out)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
print type(rounded)
print len(rounded)

a=np.matrix(rounded)
id_pred=np.matrix(id_pred)
print a.shape
print type(a)
print id_pred.shape
print type(id_pred)

output=np.concatenate((id_pred.T,a.T),axis=1)

print output
print output.shape

output=np.asarray(output)

print output.shape
print type(output)


np.savetxt("predictions_2.csv",output,header="id,salary",delimiter=",",comments='',fmt='%i')


model=svm.SVC(kernel='linear',C=1,gamma=1)
model.fit(outdata,y)
model.score(outdata,y)
#Predict Output
predicted= model.predict(pred_out)

print predicted

a=np.matrix(predicted)
id_pred=np.matrix(id_pred)
print a.shape
print type(a)
print id_pred.shape
print type(id_pred)

output=np.concatenate((id_pred.T,a.T),axis=1)

print output
print output.shape

output=np.asarray(output)

print output.shape
print type(output)


np.savetxt("predictions_3.csv",output,header="id,salary",delimiter=",",comments='',fmt='%i')