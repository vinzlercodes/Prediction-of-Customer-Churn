#plotting a KDE PLOT which is used to  for visualizing the Probability Density of a continuous variable

facet = sns.FacetGrid(df, hue="churn",aspect=3)
facet.map(sns.kdeplot,"balance",shade= True)
facet.set(xlim=(0, df["balance"].max()))
facet.add_legend()
plt.show()
