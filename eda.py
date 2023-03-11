import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

gamestop = pd.read_csv('gamestop.csv', index_col=0)
tesla = pd.read_csv('tesla.csv', index_col=0)

print(gamestop.isnull().sum())
print(tesla.isnull().sum())

gamestop['text_length'] = gamestop['title'].apply(len)

print(gamestop.head())
sns.histplot(data=gamestop, x = 'text_length')
plt.show()