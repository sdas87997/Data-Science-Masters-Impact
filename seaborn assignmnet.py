#!/usr/bin/env python
# coding: utf-8

# Q5 -->  Use the "iris" dataset from seaborn to plot a pair plot. Use the hue parameter for the "species" column 
# of the iris dataset.

# In[9]:


import seaborn as sn
iris = sn.load_dataset("iris")
sn.pairplot(iris, hue="species")


#  q6>Use the "flights" dataset from seaborn to plot a heatmap

# In[41]:


import seaborn as sn
import numpy as np

flights = sn.load_dataset("flights")
arr_2d = np.array((flights.passengers,flights.year))
print(arr_2d)
print(len(arr_2d))
#arra_organised  = arr_2d.reshape((len(flights.year),len(flights.passengers)))
sn.heatmap(arr_2d,vmin = 1949 ,vmax = 1960,cmap = "tab20")


# Que 4: Use the "diamonds" dataset from seaborn to plot a histogram for the 'price' column. Use the hue 
# parameter for the 'cut' column of the diamonds dataset.
# 

# In[49]:


import seaborn as sn

diamons_data = sn.load_dataset("diamonds")
sn.histplot( data=diamons_data,
    x=diamons_data.price,
    hue="cut" )



# 
# Que 3: Load the "titanic" dataset using the load_dataset function of seaborn. Plot two box plots using x = 
# 'pclass', y = 'age' and y = 'fare'.
# 
# Note:  pclass, age, and fare are columns in the titanic dataset.

# In[53]:


import seaborn as sn

titanic_data  =  sn.load_dataset("titanic")
sn.boxplot( x = titanic_data['pclass'], y = titanic_data['age'] )


# In[55]:


sn.boxplot( x = titanic_data['pclass'], y = titanic_data['fare']  )


# Que 2: Load the "fmri" dataset using the load_dataset function of seaborn. Plot a line plot using x = 
# "timepoint" and y = "signal" for different events and regions. 
# 
# Note:  timepoint, signal, event, and region are columns in the fmri dataset.
# 

# In[62]:


import seaborn as sn

fmri_data  =  sn.load_dataset("fmri")
sn.lineplot(
    data = fmri_data,
    x = fmri_data.timepoint,
    y = fmri_data.signal,
    hue ="region",
    style="event"
     )



# Q1> Name any five plots that we can plot using the Seaborn library. Also, state the uses of each plot
# 
# 5 plots:-
# scatterplot --- used for corelation
# displot or histplot -> distribution frquency wise
# catrplot ----------> categorical distribution
# relplot-----> various relatioanl data
# barplot----> to draw bargraphs
# jointplot -> histogram + scatter plot of data variables
# pair plot ---> jus pass data set, it will  generate maps taking 2 variables as pair

# In[ ]:




