import sklearn.datasets
diabetes_data = sklearn.datasets.load_diabetes()
print(dir(diabetes_data))
print(diabetes_data.DESCR)
print(type(diabetes_data.data))
from numpy import shape
print(shape(diabetes_data.data))
print(diabetes_data.feature_names)
print(type(diabetes_data.feature_names))
print(diabetes_data.data)
import pandas as pd 
df=pd.DataFrame(data=diabetes_data.data,columns=(diabetes_data.feature_names))
print(df.head())
import matplotlib.pyplot as plt
for feature_name in diabetes_data.feature_names:
    plt.scatter(df[feature_name],diabetes_data.target)
    plt.ylabel('Result',size=15)
    plt.xlabel(feature_name,size=15)
    plt.savefig(feature_name+".png")
    plt.show()
    df['Result']=diabetes_data.target
    print(df.corr())
    from numpy import polyfit,polyval
    age=df['age']
    result=df['Result']
    p=polyfit(age,result,1)
    plt.plot(age,result,'o')
    plt.plot(sorted(age),polyval(p,sorted(age)),'-')
    plt.show()
    corr=df.corr()
    







