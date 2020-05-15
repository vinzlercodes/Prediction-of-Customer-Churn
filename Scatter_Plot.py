#plotting scatter plots

_, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(x = "balance", y = "age", data = df, hue="churn", ax = ax[0])
sns.scatterplot(x = "balance", y = "credit_score", data = df, hue="churn", ax = ax[1])
