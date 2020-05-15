#Printing the confusion matrix
print("confusion matrix")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
print(cm)
cm_df = pd.DataFrame(cm,index=["No Churn","Churn"],columns=["No Churn","Churn"])
print("\n")
