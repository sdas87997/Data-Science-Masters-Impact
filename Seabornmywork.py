#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sn


# In[3]:


iris = sn.load_dataset("iris")


# In[4]:


iris


# In[7]:


sn.scatterplot(x= iris.sepal_length,y = iris.petal_length)


# In[9]:


sn.displot(iris["species"])


# In[11]:


sn.histplot(iris.petal_width)


# In[12]:


tips = sn.load_dataset("tips")


# In[13]:


tips


# In[17]:


tips["smoker"].value_count()


# In[22]:


sn.relplot(x =tips.tip,y = tips.total_bill,data= tips,style="sex",hue= "smoker" )


# In[25]:


sn.catplot(x = tips.day,y = tips.total_bill,data= tips )


# In[26]:


sn.jointplot(x = tips.day,y = tips.total_bill)


# In[27]:


sn.pairplot(iris)


# In[28]:


sn.pairplot(tips)


# In[31]:


sn.barplot(x= tips.day,y = tips.total_bill)


# In[34]:


sn.boxenplot(tips.tip)


# In[36]:


sn.violinplot(tips.total_bill)


# In[37]:


sn.lineplot(x= iris.sepal_length,y = iris.petal_length)


# In[38]:


sn.kdeplot(iris.sepal_length)


# #heatmap
# Heatmap is defined as a graphical representation of data using colors to visualize the value of the matrix. In this, to represent more common values or higher activities brighter colors basically reddish colors are used and to represent less common or activity values, darker colors are preferred. Heatmap is also defined by the name of the shading matrix. Heatmaps in Seaborn can be plotted by using the seaborn.heatmap() function.
# 
# seaborn.heatmap()
# Syntax: seaborn.heatmap(data, *, vmin=None, vmax=None, cmap=None, center=None, annot_kws=None, linewidths=0, linecolor=’white’, cbar=True, **kwargs)
# 
# Important Parameters:
# 
# data: 2D dataset that can be coerced into an ndarray.
# vmin, vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
# cmap: The mapping from data values to color space.
# center: The value at which to center the colormap when plotting divergent data.
# annot: If True, write the data value in each cell.
# fmt: String formatting code to use when adding annotations.
# linewidths: Width of the lines that will divide each cell.
# linecolor: Color of the lines that will divide each cell.
# cbar: Whether to draw a colorbar.
# 
# 

# In[47]:


# importing the modules
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

# generating 2-D 10x10 matrix of random numbers
# from 1 to 100
data = np.random.randint(low = 1,
						high = 100,
						size = (10, 10))
print("The data to be plotted:\n")
print(data)

# plotting the heatmap
hm = sn.heatmap(data = data)



# In[51]:


import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
  
# generating 2-D 10x10 matrix of random numbers
# from 1 to 100
data = np.random.randint(low=1,
                         high=100,
                         size=(10, 10))
  
# setting the parameter values
vmin = 30
vmax = 70
#If we set the vmin value to 30 and the vmax value to 70, then only the cells with values between 30 and 70 will be displayed. This is called anchoring the colormap.
  
# plotting the heatmap
hm = sn.heatmap(data=data,
                vmin=vmin,
                vmax=vmax,
                annot= True
               )
  
# displaying the plotted heatmap
plt.show()
#If we set the vmin value to 30 and the vmax value to 70, then only the cells with values between 30 and 70 will be displayed. This is called anchoring the colormap.


# In[53]:


# importing the modules
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

# generating 2-D 10x10 matrix of random numbers
# from 1 to 100
data = np.random.randint(low=1,
						high=100,
						size=(10, 10))

# setting the parameter values
cmap = "tab20"

# plotting the heatmap
hm = sn.heatmap(data=data,
				cmap=cmap)

# displaying the plotted heatmap
plt.show()


# In[ ]:




