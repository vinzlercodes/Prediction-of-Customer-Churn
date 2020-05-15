# plotting counter plots
#prints 7 graphs towards all the mentioned features

_, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.countplot(x = "products_number", hue="churn", data = df, ax= ax[0])
sns.countplot(x = "estimated_salary", hue ="churn", data = df, ax = ax[1])
sns.countplot(x = "tenure", hue="churn", data = df, ax = ax[2])

fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='country', hue = 'churn',data = df, ax=axarr[0][0])
sns.countplot(x='gender', hue = 'churn',data = df, ax=axarr[0][1])
sns.countplot(x='credit_card', hue = 'churn',data = df, ax=axarr[1][0])
sns.countplot(x='active_member', hue = 'churn',data = df, ax=axarr[1][1])
