import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/spam.csv",encoding="latin-1")

df= df[["v1","v2"]]

df.columns=["label","message"]  

print(df.head())

sns.countplot(x="label",data=df)

plt.title("spam vs ham messages")

plt.savefig("visuals/spam_vs_ham.png")


plt.show()



df['message_length']=df['message'].apply(len)

sns.histplot(data=df,x='message_length',hue='label',bins=50)

plt.title("Message Length Distribution")    
plt.savefig("visuals/message_length_distribution.png")
plt.show()




