import numpy as np 

w1=np.zeros((108,50))
w1=np.asmatrix(w1)
w2=np.zeros((50,1))
w2=np.asmatrix(w2)
fil=open("k.txt",'w')
np.savetxt(fil,w1,delimiter=",")
np.savetxt(fil,w2,delimiter=",")
fil.close()

