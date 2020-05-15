#Printing the confusion matrix
print("confusion matrix")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
print(cm)
cm_df = pd.DataFrame(cm,index=["No Churn","Churn"],columns=["No Churn","Churn"])
print("\n")

#visually representing the matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm_df,annot=True ,cmap="Blues",fmt="g" )
plt.title("Confusion Matrix",y=1.1)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

print(((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% overall ACCURACY of trained model on the dataset')
print("\n")
