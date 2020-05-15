Churn Prediction in the Financial Sector using Artificial Neural Networks 

Abstract 
Customers are the foundation on which most organizations and businesses operate their services. With the ever growing number of customers, comes the responsibility of making sure of customer loyalty and satisfactory customer services. A way to make sure this happens is by maintaining and analyzing the plethora of data generated daily. To gain a competitive edge in the industry, organizations must use this data to their advantage and act quickly to retain customers who may be considering stopping their services. The following research analyzes a dataset from the banking industry to predict the churn rate of current and past customers the bank has had. An Artificial Neural Network model programmed in Python is utilized to carry out the predictions and the accuracy is calculated. The experimental results, proportion of customers that have exited and retained, probability density of bank balance, correlation between attributes, top three factors responsible for churn, are all plotted in the form of a variety of graphs. Python libraries Matplotlib and Seaborn were used for creating the plots. Furthermore, the confusion matrix is calculated along with plotting graphs for the experimental results. Through this research we aim to highlight a model which can be utilized by financial organizations to better track their customer churn patterns. 

Keywords:
Finance, Churn, Customer, Neural Network, Prediction, Python






I. Introduction
In today’s modern Fin-Tech and Consumer-Centric industry the most valuable asset is the customer. With the rise of technology virtually breaking the chain of physical monetary transactions and innovations giving rise to not just competitions but to security concerns such as frauds. Churn Rates (customers leaving or closing accounts) in companies for various reasons have also as result become a rising concern. Hence, the retention of customers has become paramount. According to the Qualtrics Banking Report, “poor service” was the reason the customers left. 56% of these customers stated that with better services they could have been retained. According to some customers they felt unvalued by the banks themselves. In a customer centric industry, customer loyalty is the decider of all, and customer experience invites loyalty. Research at Bain & Company suggested that a 5% decrease in customer population can reduce profits from 95% to 25%. Therefore, today churns have become a bitter outcome to prevent heavy losses and calls for deep understanding of the customer’s needs, preferences, sentiments and behaviour.
Humongous amount of information is generated out of these organisations everyday. Companies can utilise the data to gain deep insights about their customers. Studying transaction patterns and user data, firms can integrate them with predictive analytics to detect and even foresee credit card frauds and flag money laundering techniques. Financial data can help recognise stock price patterns and even help manage risk assessments. Sentimental Analysis of customer feedback will help organisations cater to the customer experience much more accurately. This will result in more profits and revenue as well as efficient usage of resources. Using the right factors organisations will be able predict potential churns. Detection of these churns can help companies stop them by acting accordingly with card offers, discount on annual fees, or lower loan rates.
The most common strategies used for prediction are Decision Tree (DT), Logistic Regression (LR), Support Vector Machine (SVM) and Neural Networks (NN). In addition, the Decision tree is used for resolution of Classification Problems to divide the data into two or more than two classes. Logistic regression gives the probability via presenting input/output fields and set of equations and elements causing customer churn. Finally, it's been found that Deep Learning methods (Neural Networks) have the most accurate prediction outputs. Hence, Artificial Neural Network (ANN) has been adopted since it gives us with the right activation function most desirable probabilistic insights.



II. Literature Survey 
There is no doubt that the churn rate can damage a company's revenues and based on this understanding there is a need to analyse various  techniques that are used for churn prediction and customer retention. One such technique is the Artificial Neural Network (ANN) which is a popular method used for churn prediction.[13]  Multi-Layer-perceptron method is used which is trained using the Backpropagation Algorithm which is a feed forward model using  supervised learning. Results have shown that the Neural network based model shows better performance compared to Logistic Regression with an accuracy of 94% and a F-Measure of 77%.[1] When we talk about Client Stir which means that the client leaves the management of one company for another company , we need to keep in mind that the expense of getting a new client is thrice than that of holding one. In such cases Decision Trees are more generally used for the prediction of future occasions.[16] When considered the Bank informational Index, recall and precision were  the two factors used to judge the performance of the Decision Tree model Neural Network model and the Support Vector Machine (SVM) model . The precision of the Decision Tree system was 85.66% ,neural network system was 82.3% where as the SVM was only 73% while the recall being 92% , 89%, and 84% respectively which shows that when talking about Client Stir,Decision Tree and Neural Network models are highly recommended.[2]
Monetary accounts such as an Investment Account where customers are able to make purchases and multiple trades, when closed result in the ultimate failure of the banks as they will be unable to provide cash for loan requirements. Thereby not earning the interest which keeps these banks afloat. The results of the K-Nearest Neighbour Classification Algorithm were based on criterias like True Positive Rate (TP  Rate),False Positive Rate(FP Rate)and Confusion matrix. It was predicted that 28.17% of the people closing investment account were Deputy Bureaucrats(females aged between 41-50 years of age) and Lawyers(aged between 51-60) when the socio-demographic data of the customers were analysed.[3] To find patterns  that are hidden, big data analytics is a great tool . It can also be used to find unknown correlations and various trends in the market.[12] A Targeted Proactive Retention model helps in such cases where we can predict the churn in advance and also look for the customer retention. A four step implementation technique was applied which involved data preparation, model creation,model deployment and advanced preparation. A predictor was designed using a Naive Bayes and Artificial Neural Network Model and results showed an estimate of the probability of churners was obtained by combining the output of three graphs namely Accuracy, Regulation and Duration which implied  that the steeper the graph, the better was the accuracy, similarly we have to find a threshold value from the list of probabilistics results to predict is a customer is a churn or not.[4]

A similar approach using ANN was implemented to predict customer retention rate with an accuracy of upto 86 percent [7]. This approach passed a dataset with fields such as credit score, geography , gender, age, tenure, balance through multiple layers of neural network resulting in an output that demonstrated the predicted values. To correct errors, back propagation was applied. Customer Churn is most commonly predicted using Decision tree and logistic regression, however, one such novel approach employed the use of LLM (Logic Leaf Model) to segment and predict the data [6]. The first step involves creation of a decision tree followed by applying logistic regression on each of the segments. This approach allows for accurate models and offsets the need for tradeoff between predictive performance and comprehensibility.[18] It is important to continually test out various algorithms to analyze the vast amount of data generated from customers in the most accurate manner. Boosting of Support Vector Machine is another such learning method which leads to more accurate predictions. The dataset is initially preprocessed by applying techniques such as feature attribute selection to decrease the dimensionality. It is then passed through a Support Vector Machine Recursive Feature process Elimination (SVM-RFE) to group the data where positioning score is calculated for each of the arrangements [5]. Finally, the customer information is passed to a MATLAB simulation where the F1-measure, precision and accuracy was calculated and noted to have higher classification accuracy where SVM gave a classification accuracy of 99.533. [20]

An algorithm based on the prefix tree method called FP Growth method was tested on a retail marketing company’s dataset. This method used association rule mining to find the number of customer churns.[17] Association rule mining is a method that can extract correlations and patterns from large sets of data easily . Overall, it is a two step process, the first step involves finding all frequent itemsets, this can be done using the Apriori or FP Growth method. The second step is to generate strong association rules from the frequent item sets [8]. The rules must always satisfy predefined support and confidence for data given.  It is most commonly used in areas such as telecommunication industry and market for risk management, inventory control, among other use cases. Although many research papers utilized ANN models to predict churn, there are others that have used techniques such as profit driven decision trees for churn prediction. A novel approach used a technique called ProfTree, which is an algorithm to learn decision trees, it was observed that it performed better than 6 classic tree models and gave better accuracy [9]. Another approach utilized a feature selection model with simulated annealing and particle swarm optimization. It uses three different types of PSO and compares the results with classic models such as K-nearest neighbour, random forest and support vector machines. It was noted that the accuracy and performance was very high using the feature selection model when the dataset was imbalanced [10].

Artificial Neural Networks have become a go-to solution when it comes to classification and prediction-based problems that require decision-making and optimizations. The reason why they have become so prominent as compared to other statistical methods is because they are better at mapping inputs and outputs of the model without the hassle of complex formulae or even relationships between them. Hence, an MLP Model based ANN is optimal for churn prediction of customers from a healthy dataset from the CRM.[11] Another approach that has been developed to tackle customer Churn Prediction involves the simulation of different Machine Learning Classifier techniques. The classifiers that were used were: KNN Classifier, Random Forest Classifier (RF) and XGBoost classifier. Each classifier features and prediction techniques were evaluated using the univariate analysis method. F-score and Accuracy scores were the final metric values by which each classifier's performance were evaluated by. It came out that the XGBoost classifier outperformed its competition significantly in customer Churn Prediction. Whilst KNN received a F-score of 0.495 and Accuracy score of 0.754. RF received an F-score of 0.506 and an Accuracy score of 0.775. In the end XGBoost got the highest of both with an F-score of 0.582 and an Accuracy score of 0.798.[14]































III. DataSet 

For the Assignment, a dataset containing 10000 rows and 14 columns was considered. This dataset was obtained from Kaggle.  This dataset contains information about the card holder in a bank and their possibility of churn. The attributes in the dataset are customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, and churn.  A value of 1 for churn indicates that the member has exited while 0 indicates the member is still using the bank. 

Below is a tabular representation of the dataset contents:


Table 1:  Description of Dataset

Taking a look at few of the rows of the dataset, the contents of each column are as as below:


Table 2:  Sample Dataset
Table 3 contains all unique values that the country column takes:

Table 3:  Description of country
Table 4 contains values the credit_card column takes and the meaning associated with it:

Table 4: Description of credit_card
Table 5 contains values that active_member column takes and the meaning associated with it:

Table 5: Description of active_member
We have ignored 2 columns from the main dataset, namely row number and surname as they were not required for accurate model predictions. The credit score column represents the creditworthiness of a person based on the past data. The product number column indicates the number of accounts a member has at that bank. Apart from that, the credit card column indicates if the person has a credit card with the bank or not. The active member column indicates if a member is active or not based on the criteria set by the bank. 

Overall, the dataset contains a wide variety of values for columns, with details about users from 3 main countries: France, Spain & Germany. Moreover, there are no blank or incomplete rows in the dataset which allowed us to efficiently use the dataset for our models. After applying the models on a training set of data, a value for churn of each customer can be predicted for future data that is tested.











IV. Analysis of Applied Methodology

As we have established before that customer churn is a serious threat to the lucrativity and success of financial businesses in the consumer-driven industry. Today leading consultancies and fin-tech companies are investing heavy research and resources into tackling this issue in terms of both prevention and the cure. Our research dwelves into tackling churn in the prevention process, where we are predicting the probability of a current customer churn based on their overall financial records. The methodology is two-folded firstly we create a prediction model and test it and lastly, conduct an analysis of the various attributes of customers that contribute to churn today. 


The statistical results collected from the Kaggle Customer Churn Data have the following non-numeric data attributes:



Table 6: Non-numeric Data Attributes Description




The table below contains an account of the numeric data attributes of the dataset:



Table 7: Numeric Data Attributes Description



IV.I Analysis of the Neural Prediction Model

In this research an Artificial Neural Network model has been developed as the prediction model for customer churn in the financial industry. The model makes use of Multiclass Classification which consists of Softmax Regression, which in turn amplifies the concept of Logistic Regression method of classification. 

IV.I.I Logistic Regression

The logistic Regression is similar to its Linear counterpart and offers a solution to Binary Classification problems, with the output being between the values of 0 and 1. And also each class is not required to have equal values of variance or normal distribution. But unlike Linear Regression, Logistic Regression  has a more complex cost function called ‘Sigmoid Function’ for accurate mapping of probability values. 



Fig. 1: Linear Regression vs Logistic Regression

We start by noting the linear regression equation (hypothesis) and we will work on it to make it a logistic regression function (1):

                              


A few adjustments are made to the previous equation with respect to the logistic regression concept (2):

 = 


We then make use of the Sigmoid function:


We add it to equation (2) to get values between 0 and 1:




The equation then looks like below (3):



Hence, the final Logistic Regression function looks like this:


Fig. 2: Logistic Regression  Function 

This very classifier function can now handle any number of categorical or/and numeric values/variables.












IV.I.II Artificial Neural Network

In this chapter we will delve deep into the final Artificial Neural Network Model, discussing how the model structure has been built, which activation function we have utilised, the forward and back propagation of the model, the gradient descent method has been used, the cost function and finally the kernel initialiser utilised.

The model consists of 2 Hidden Layers and each hidden layer has a Binary Classifier (Logistic Regression) for the Churn Rate prediction. The Input and Output layers have the Activation Functions of Relu for the node and weights initialization. The Gradient Descent has been optimised with Adam and Binary Cross Entropy is our Loss Function.


IV.I.II.I The Model Architecture

The Artificial or Feedforward Neural Network (ANN) is a multi-layered fully connected neural network.It consists of an input layer, 2 hidden layers, and an output layer. Every node in one layer is connected to every other node in the next layer. Below is a diagrammatic representation of our ANN model:



Fig. 3: Artificial Neural Network Model









Diving deeper into our model, we then examine the nodes of either the hidden or output layer, and we come across the following:




Fig. 4: Node Structure in the Hidden/Output Layers

In the above node structure the value of z can computed using the following equation below (5):




We then proceed to simplify the equation by removing the Bias term since it is always valued 1 for all input nodes (6):



In summary, each layer of the ANN performs a non-linear transformation of the input from one vector space to the other. 





The different nodes structures of the various model layers (input, hidden and output) can be viewed in their vector forms in the figure shown below:




Fig. 5: Vectorised forms of the different ANN model layers




















IV.I.II.II Activation Function

Activation function is crucial to our ANN model since this is what is responsible for the nonlinear functional mappings between the inputs and output variables. Four our model we have made use of the most sought after activation function in neural networks for its efficiency, that is the Rectifier Function (Relu). The great thing about this function is that it does not require normalisation and returns a value of “yes” or “no” in percentage form. 



Fig. 6: Rectifier Function



















IV.I.II.III Forward and Back Propagation

Forward Propagation (FP) works from left to right of the model, the neurons (nodes) are activated by the Relu activation function in a way that each neuron’s activation is limited by the weights (defined by the model itself). The activations are propagated through until the predicted result is obtained according to the function.

The predicted result (which is calculated in FP) is compared to the actual result (in the dataset) and the error is generated based on their difference.

In contrast the Back Propagation (BP) as the name suggests is its exact opposite, occurring from right to left, the error generated before is back propagated. The weights are updated according to how much they are responsible for the error and updated to minimise the error.

1 full iteration of the forward propagation, error generation and back propagation followed by the weights being updated is known as an Epoch.

The math behind how each propagation treats the functions is shown below:
.
Fig. 7: Forward and Back Propagation










IV.I.II.IV Stochastic Gradient Descent using Adam

Stochastic Gradient descent is an optimization method for finding the minimum of a function. Starting from a random value, from the current point, steps in the opposite direction of the gradient to reach the global minimum of the entire function by means of differentiation. In our model we make use of Adam instead of the classic approach. Since in the original method the learning rate remains constant throughout all parameters the same is not with Adam. Adam maintains a different learning rate per parameter and also adaptes them based on the average of recent values of the gradients for the weights. 



Fig. 8: Adam Optimizer (top) vs Stochastic Gradient Descent (bottom)












IV.I.II.V Cost Function

The cost function that we have adopted for our model is the Binary Cross-Entropy function. It is essentially the difference between 2 probability distributions. In our case the actual value distribution and the predicted value by our activation function.

Let our 2 probability distributions be P and Q. Then cross-entropy is written as H(P,Q). Here P is the actual value distribution and Q is the predicted values distribution.

Hence, the final equation (7) is: H(P, Q) = – sum x in X P(x) * log(Q(x))

P(x) is the probability of the variable/class x in P, Q(x) is the variable/class x in Q and log is the base-2 logarithm.

The Loss Function graph generated for the performance of our model has been further discussed in the experiment result analysis chapter.


IV.II Performance Evaluation of the Model

Fig. 9: Proposed Model Diagram
V. Experimental Results

The results have been divided into four sections : visualisation for different attributes, accuracy and loss models followed by the confusion matrix and finally the ROC curve.

V.I Visualization

We begin analyzing the experimental results by visually comprehending the various attributes that contribute to customer churn. The figures below depict the count plots for various numeric attributes.

Fig 10: Count Plots showing the experimental results


We can infer the following things from the above images:

Out of the three countries whose data was collected, Germany had the maximum churn rate compared to France and Spain.
When it comes to Gender, we can see that overall Females have a greater churn capacity than Males.
We can also see that the people who have a credit card have a greater churn rate than the people who don’t have a credit card.
Finally we see that those members who are active stay with the bank and don’t churn as much as compared to those members who are inactive.

The following Pie Chart will give a greater insight on the total number of people who have exited and those who have been retained by the bank.


Fig 11: Pie Chart showing the proportion of customers Exited and Retained



As we can see from the above pie chart,the bank was able to retain 79.6% of the customers while 20.4% did exit the bank.

Next, we have also plotted a KDE plot which is used for visualizing the Probability Density of a continuous variable, in our case the Balance amount of the customers in the bank.



Fig 12: KDE plot to visualise the probability density of the bank balance

The major reason for the churn was attributed to the Balance of the customers as shown in the above image. The Gaussian distribution shows that most of the customers with the Balance between 8000 and 16,000 succumbed to churn while the peak was observed at 12500.

To understand the relationship better among all the variables a heatmap was plotted as shown below:


Fig 13: Heatmap showing the correlation between the attributes

In a heat map, each square shows the correlation between the variables on each axis. Values closer to zero means there is no linear correlation between the two variables for example between balance and credit_score , estimated_salary and tenure etc. The values that are close to 1 are more positively correlated showcasing good dependency on each other for example the age and churn and balance and churn etc.  A correlation closer to -1 indicates that the two variables are inversely proportional like the balance and product_number , active_member and churn etc.For the rest the larger the number and darker the colour the higher the correlation between the two variables. 


The following table shows the top three variables contributing to churn namely Age, Balance and Estimated_Salary whose inference was drawn from the heatmap.


               
 Table 8:  The top three variables contributing to churn


















We can also show the graphical representation of the above table with the following box plots.





      Fig 14: Box Plots showing the top three variables contributing to churn







We can further study the relationship between the top two variables contributing to churn namely Age and Balance with the help of a scatter plot as shown below.




Fig 15: Scatter plot showing the relationship between Age and Balance
We can infer from the above scatter plot that the range of the majority churn age of various customers of the bank lie between 42 and 60 and the range of most of the customer’s balances were between 5000 and 150000.















V.II Accuracy and Loss Models
Each entry of our dataset has gone through 100 epochs of training in the Artificial Neural Network Model. A snippet of all the epochs is shown below.





Fig 16 : 100 epochs of training in ANN
From the above shown epochs we are able to get the accuracy as well as the loss with each dataset which is shown as a graphical representation below.

Fig 17: Accuracy of our Model
As we can infer from the above line chart, our model hits an overall of 84.5 % accuracy on the ability to predict the probability that a client will churn from the bank or not from the dataset.
The loss of our model is around 0.36 as shown in the below line chart. Unlike accuracy, loss is not a percentage. It is a summation of the errors made for each example in the training or validation set. 


Fig 18: Loss of our Model


V.III Confusion Matrix
The below image shows the graphical representation of a confusion matrix:

Fig. 19: Confusion Matrix 
As we can see, out of the 2000 datasets that were used for the testing purpose, our model was able to accurately predict 1556 +134 = 1590 correct predictions and 271+39 =  410 incorrect predictions. Following Table also highlights various other inferences from the confusion matrix.



 Table 9:  Statistics to interpret the confusion Matrix
V.IV ROC Curve
In Data Science , performance measurement is a crucial task therefore when it comes to classification problems we can count on an AUC-ROC curve. ROC is a probability curve and AUC talks about the degree of separability which tells us how accurately our model is capable of distinguishing between classes.
The ROC curve is plotted with the True Positive Rate on the y-axis and the False Positive Rate along the x-axis. The AUC(Area under Curve) values lies between 0.5 to 1, where 0.5 denotes an underperforming classifier and 1 denotes a very strong classifier. As we can see from the below ROC curve, our model AUC has a value of 0.80 which means that there is 80% chance that our model will be able to distinguish between the churn and the no churn classes.


Fig 20: ROC Curve







V.VI Analysis for Churn Contribution Attributes
The most important part in customer churn management is planning effective and durable retention strategies. Based on our experimental analysis it has become evident that there are 3 major attributes contributing to customer churn are Age, Balance and Estimated Salary. 
Age has the highest impact on customer churn, and the churn is more among females aged 41 to 50 than male. This could be due to a multitude of underlying reasons. One could be customer dissatisfaction when it comes to service usage. Another reason could be that, under a study conducted at IMF (International Monetary Fund) is an evident gender gap in account holding. To be able to tackle these situations banks can offer special financial policies and advantages to female bank account holders for example better credit scores and tax privileges and also perks in the form of numerous retail offers and discounts on medical and rationing services exclusive to female account holders since middle age women often tend to give medical benefits and everyday rationing and bill paying more importance. This will not only give confidence in more female account holders and also give current holders impetus to stay with respect to the added benefits and services and rewards system.
Balance is the second most contributing factor towards customers leaving the bank. There are  a number of possible reasons as to why this could be, the customer is not happy with the interest rates offered by the account and is offered a better rate from a competitor. Another reason that balance is often an issue is due to the policy of the accounts having to hold a certain value of money to remain active else penalties are raised on the accounts, often driving out customers who may be financially not as strong or facing certain fiscal difficulties. Solutions could include having better interest rates on loyal customer accounts for better retention. As well as having consensus for those facing financial difficulties and offering leniency on the personalities and minimum bank balance amounts according to the situation with proof. 
Lastly, Salary was found to be the final churn contributing attribute. It is commonly noted that people whose salaries are on the lower side of the spectrum often churn as there is no potential promotion in terms of increment or job growth which may result in their closing of their bank account in general. When we talk about people whose salaries are on the higher range of the spectrum, their churn factors could be better interest rates offered by other competitor banks and insurance companies along with the added benefits they get like the car and home insurances for longer periods of time or the entails of the insurances which in turn causes more inclinations towards these banks and result in the customer churn. It is said that the company’s most important customers are its own employees, hence, offering promotions and better financial return on their performances will not only increase their performance and profits but also maintain healthy retention due to recognition of loyalty and efforts. Another solution includes insurance and other financial enterprises to make sure they keep an eye for competition and also customer requirements and wants. Upgrades in insurance rates and also interest rates on their accounts as discussed before will definitely flatten the churn curve and raise profits.





















VI. Conclusion and Future Work
In this study, we have utilized a dataset with over 10000 rows and used 8000 rows as the training set for the model. Following that, the model was applied on 2000 test rows of the dataset to predict customer churn with 84.5% accuracy. We used an Artificial Neural Network model to perform these predictions. From the resultant data, we not only predicted the accuracy but also created a confusion matrix and plotted the ROC Curve to study the results further. It was further observed that the attributes age, balance and estimated_salary contributed heavily to the resultant value of churn. Graphs such as heat maps, scatter plots and box plots were plotted to study correlation between various attributes. These were plotted using the plotting libraries available in Python. Plotting libraries have been used to present research results in an easy manner for different kinds of datasets by many researchers. Such libraries have been used heavily in the field of geospatial data analysis to create 3D comparative plots and other visualizations [12].

In this paper, we have only applied an ANN model approach to predict the results. In the future we plan to explore other data science and machine learning models to generate more accurate predictions [13]. Furthermore, we will also expand the application of our model to generate customer churn predictions for different industries such as retail, online dating [14] and the telecommunication industry[15]. The telecommunication industry is another industry that highly uses customer churn models to retain its customers [16]. In some cases, the datasets might have a class imbalance, in such cases the XGBoost model gives the best accuracy as per previous research [17].

We also plan to  implement models that will allow us to suggest actions that an organization can undertake to retain customers on the basis of strategies that have worked for other companies in the past [18]. In other research papers, methods such as C4.5 decision tree [19] and Multi-objective Meta-Analytic Methods [20] have been used and we plan to incorporate learnings from these models to improve our accuracy even further in our future work. 


References 
[1] G. Ravi Kumar, K Tirupathaiah, B Krishna Reddy,"Client Churn Prediction of Banking and fund industry utilizing Machine Learning Techniques' ',JCSE International Journal of Computer Science and Engineering,2018.
[2]Praveen Asthana,"A Comparison of Machine Learning Techniques for customer churn prediction",International Journal of Pure and Applied Mathematics,2018.
[3] Fatih CIL,Tashin Cetiyokus, Hadi Gokcen,"Knowledge Discovery on Investment Fund Transactions Histories and Socio - Demographic Characteristics for Customer Churn",International Journal of Intelligent Systems and Applications in Engineering,2018.
[4] B. Mishachandar , Kalkeli Anil Kumar,"Predicting Customer Churn using Proactive Retension",International Journal of Engineering and Technology,2018.
[5] Saran Kumar, S.Viswanandhne , S. Balakrishnan,"Optimal Customer Churn Prediction System using Boosted Support Vector Machine",International Journal of Pure and Applied Mathematics,2018.
[6] Arno De Caigny, Kristof Coussement, Koen W. De Bock,"A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees",European Journal of Operational Research,2018.
[7] V. Sahaya Sakila, Ishan Kanungo,Anson Joel,"Prediction of Bank’s Customer Retaining Rate Using Neural Networks",International Journal Of Research in Engineering, Science and Management,2018.
[8] Mie Mie Aung, Thae Thae Han,Su Mon Ko,"Customer Churn Prediction using Association Rule Mining",International Journal of Trend in Scientific Research and Developement(IJTSRD),2019.
[9] Höppner, Sebastiaan & Stripling, Eugen & Baesens, Bart & Broucke, Seppe vanden & Verdonck, Tim, “Profit driven decision trees for churn prediction”,  European Journal of Operational Research, Elsevier, Vol. 284(3), 2020
[10] J. Vijaya & E. Sivasankar, “An efficient system for customer churn prediction through particle swarm optimization based feature selection model with simulated annealing”, The Journal of Networks, Software Tools and Applications, 2018
[11] Seyed Hossein Iranmanesh, Mahdi Hamid, Mahdi Bastan, Hamed Shakouri G., Mohammad Mahdi Nasiri ,"Customer Churn Prediction Using Artificial Neural Network: An Analytical CRM Application",Proceedings of the International Journal on Industrial Engineering and Operations Management Pilsen, Czech Republic, 2019
[12] J. Pamina, J. Beschi Raja, S. Sathya Bama, S. Soundarya, M.S. Sruthi, S. Kiruthika, V.J. Aiswaryadevi, G. Priyanka, “An Effective Classifier for Predicting Churn in Telecommunication”, Journal of Advanced Research in Dynamical & Control Systems, Vol. 11, 01-Special Issue, 2019
[13] Polina LEMENKOVA , “Generic Mapping Tools and Matplotlib Package of Python for Geospatial Data Analysis in Marine Geology”, International Journal of Environment and Geoinformatics, Volume 6, Issue 3, 2012

[14]  Andrea Dechant, Martin Spann, Jan U. Becker, “Positive Customer Churn: An Application to Online Dating”,  Journal of Service Research, Volume 94, 2019

[15]  Adnan Amin, Feras Al-Obeidat, Babar Shah, Awais Adnan, Jonathan Loo, Sajid Anwar, “Customer churn prediction in telecommunication industry using data certainty”, Journal of Business Research, 2018

[16] Abdelrahim Kasem Ahmad, Assef Jafar & Kadan Aljoumaa, “Customer churn prediction in telecom using machine learning in big data platform”,  The Journal of Big Data, 2019

[17] Aayush Bhattarai, Elisha Shrestha, Ram Prasad Sapkota, “Customer Churn Prediction for Imbalanced Class Distribution of Data in Business Sector”, Journal of Advanced College of Engineering and Management, 2019

[18]  Jan U. Becker, Martin Spann, Christian Barott, “Impact of Proactive Post Sales Service and Cross-Selling Activities on Customer Churn and Service Calls”, Journal of Service Research, 2019 

[19] Weibin Deng, Linsen Deng, Jin Liu, Jie Qi, “Sampling method based on improved C4.5 decision tree and its application in prediction of telecom customer churn”, International Journal of Information Technology and Management, 2019 

[20] Mohammad Nazmul Haque, Natalie Jane de Vries, Pablo Moscato, “A Multi-objective Meta-Analytic Method for Customer Churn Prediction”,  The Journal of Business and Consumer Analytics: New Ideas, 2019


