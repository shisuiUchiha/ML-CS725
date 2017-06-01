

I have used cross validation to get the values of lambda and learning_rate in the code.I have taken 80% of train data and obtained the weights or coefficents. Then I used the remaining 20% percent of train data to see the closerness of my output data.By using this I have finally decided my lambda and learning rate.

Working of code:

1)I first normalised the data and used it for further calculations
2)I have used 2 functions gradient_descent and gradient_descent_p.The first one is to calculate the parameters using ridge regression and the second one is used for calculating output data for 3 different values of p i.e., 1.2 , 1.5 , 1.8 respectively


The output I got using gradient descent seems to somewhat better then using closed form solution.


  
