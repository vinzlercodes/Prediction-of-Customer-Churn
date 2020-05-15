#plotting a swarmplot
plt.figure(figsize=(8, 8))
sns.swarmplot(x = "credit_card", y = "age", data = df, hue="churn")
