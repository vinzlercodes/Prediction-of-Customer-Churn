The following research analyzes a dataset from the banking industry to predict the churn rate of current and past customers the bank has had. An **Artificial Neural Network Model** programmed in Python is utilized to carry out the predictions and the accuracy is calculated.

The experimental results, proportion of customers that have exited and retained, probability density of bank balance, correlation between attributes, top three factors responsible for churn, are all plotted in the form of a variety of graphs. 

Python libraries **Matplotlib and Seaborn** were used for creating the plots. Furthermore, the confusion matrix is calculated along with plotting graphs for the experimental results. Through this research we aim to highlight a model which can be utilized by financial organizations to better track their customer churn patterns.

**Implementation Steps**

First lets import the libraries needed for **Loading the Data and the Data Pre-Processing**

![ANN1](https://user-images.githubusercontent.com/34100245/82065238-7bfed280-96eb-11ea-9cc2-ab52d99df6c1.PNG)

We then Load and Pre-Process the Data

![ANN2](https://user-images.githubusercontent.com/34100245/82070283-25e15d80-96f2-11ea-89e3-48df1374ca91.png)

We then create our 2 Layered ANN model

![ANN3](https://user-images.githubusercontent.com/34100245/82070626-a011e200-96f2-11ea-868e-c4ef0d219a24.PNG)

Then its time to Train our model

![ANN4](https://user-images.githubusercontent.com/34100245/82071183-5f669880-96f3-11ea-9a34-75e055c29d18.PNG)

The last few epochs look like the following:

![ANN5](https://user-images.githubusercontent.com/34100245/82071316-92a92780-96f3-11ea-9902-ce78a9603a00.PNG)

Now that the model has been succesfully trained, we can print out its **Accuracy Matrix**

![ANN6](https://user-images.githubusercontent.com/34100245/82071605-f895af00-96f3-11ea-8de7-4e368de54775.PNG)

From here, now for the Analysis part one is free to visualise the data and the cooreleation between the various customer 
attributes vs churn possibility.
Out of the above list of graph codes one can copy paste the code from the respective files into the main **Ann.py** file and get the respective graphs.

If the 





If you like the work and my methodology please do leave a review and a star.

***Thank You***
