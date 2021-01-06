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

Visualising our **Confusion Matrix** by adding the **Confusion_Matrix.py** file to the main file, ist is often used to describe the performance of a classification model on a test data for which the true values are known. It allows the visualization of the performance of an algorithm:

![ANN7](https://user-images.githubusercontent.com/34100245/82073006-34317880-96f6-11ea-815a-7bde3c02412f.PNG)
![ANN8](https://user-images.githubusercontent.com/34100245/82073223-94281f00-96f6-11ea-8b6d-ed2633b01209.PNG)

Visualising our **Receiver Operating Characteristic Curve** by adding the **ROC_Curve.py** file to the main file, illustrates the capaability of a binary classifier system. It is created by plotting the true positive rate against the false positive rate. 

![ANN9](https://user-images.githubusercontent.com/34100245/82073601-1a446580-96f7-11ea-8386-4770d560cbbe.PNG)

Visualising our **Pie Chart** by adding the **Pie_Chart.py** file to the main file:

![ANN10](https://user-images.githubusercontent.com/34100245/82073847-7ad3a280-96f7-11ea-91b4-185168362de5.PNG)

Visualising our **6 Counter Plots** by adding the **Counter_Plot.py** file to the main file:

![ANN11](https://user-images.githubusercontent.com/34100245/82074435-6348e980-96f8-11ea-9d53-d70207ba1011.PNG)
![ANN12](https://user-images.githubusercontent.com/34100245/82074583-9e4b1d00-96f8-11ea-919f-bf749815bc38.PNG)

Visualising our **Scatter Plots** by adding the **Scatter_Plot.py** file to the main file. The Plots show how much one variable is affected by another.

![ANN13](https://user-images.githubusercontent.com/34100245/82075253-95a71680-96f9-11ea-997a-c50eec547900.PNG)

Visualising our **Heat Map** by adding the **Heat_Map.py** file to the main file. The Graph shows the intensity of how one variable affects the proabability of another even happening.

![ANN14](https://user-images.githubusercontent.com/34100245/82075416-d30ba400-96f9-11ea-83c1-335bfb1d4519.PNG)

Visualising our **Kernel Density Estimate** by adding the **Heat_Map.py** file to the main file,The graph is used for visualizing the Probability Density of a continuous variable.

![ANN15](https://user-images.githubusercontent.com/34100245/82075520-ffbfbb80-96f9-11ea-9675-330c0bcb46d1.PNG)

Visualising our **Swarm Plot** by adding the **Swarm_plot.py** file to the main file:

![ANN16](https://user-images.githubusercontent.com/34100245/82076583-b1132100-96fb-11ea-8def-13a5f51c4ddc.PNG)

Visualising our **Box Plot** by adding the **Box_Plot.py** file to the main file. A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”).

![ANN17](https://user-images.githubusercontent.com/34100245/82076919-24b52e00-96fc-11ea-9c2f-385f113c0993.PNG)
![ANN18](https://user-images.githubusercontent.com/34100245/82077127-870e2e80-96fc-11ea-82a8-6e5b0dd0c2a0.PNG)


---

If you do find this repository, why not give a star and even let me know about it!

Feel free to express issues and feedback as well, cheers!

















If you like the work and my methodology please do leave a review and a star.

***Thank You***
