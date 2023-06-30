#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Histogram plot
import matplotlib.pyplot as pyplot

pop = [22,55,62,45,21,22,34,42,42,4,2,8]
bins = [1,10,20,30,40,50]
pyplot.hist(pop, bins, rwidth=0.6)
pyplot.xlabel('age groups')
pyplot.ylabel('Number of people')
pyplot.title('Histogram')

# Print the chart
pyplot.show()


# In[5]:


#Area Plot
import matplotlib.pyplot as pyplot

days = [1,2,3,4,5]
age =[63, 81, 52, 22, 37]
weight =[17, 28, 72, 52, 32]

pyplot.plot([],[], color='c', label = 'Weather Predicted', linewidth=10)
pyplot.plot([],[],color = 'g', label='Weather Change happened', linewidth=5)


pyplot.stackplot(days, age, weight, colors = ['c', 'g'])
pyplot.xlabel('Fluctuation with time')
pyplot.ylabel('Days')
pyplot.title('Weather report using Area Plot')
pyplot.legend()

# Print the chart
pyplot.show()


# In[8]:


#PI Plot
import matplotlib.pyplot as pyplot

slice = [12, 25, 50, 36, 19]
activities = ['NLP','Neural Network', 'Data analytics', 'Quantum Computing', 'Machine Learning']
cols = ['r','b','c','g', 'orange']
pyplot.pie(slice,
labels =activities,
colors = cols,
startangle = 90,
shadow = True,
explode =(0,0.1,0,0,0),
autopct ='%1.1f%%')
pyplot.title('Training Subjects')

# Print the chart
pyplot.show()



# In[10]:


#Scatter plot
import matplotlib.pyplot as pyplot

x1 = [1, 2.5,3,4.5,5,6.5,7]
y1 = [1,2, 3, 2, 1, 3, 4]
x2=[8, 8.5, 9, 9.5, 10, 10.5, 11]
y2=[3,3.5, 3.7, 4,4.5, 5, 5.2]
pyplot.scatter(x1, y1, label = 'high bp low heartrate', color='r')
pyplot.scatter(x2,y2,label='low bp high heartrate',color='g')
pyplot.title('Smart Band Data Report')
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.legend()

# Print the chart
pyplot.show()


# In[13]:


#Bar plot
import matplotlib.pyplot as pyplot

pyplot.bar([0.25,2.25,3.25,5.25,7.25],[300,400,200,600,700],
label="Carpenter",color='b',width=0.5)
pyplot.bar([0.75,1.75,2.75,3.75,4.75],[50,30,20,50,60],
label="Plumber", color='g',width=.5)
pyplot.legend()
pyplot.xlabel('Days')
pyplot.ylabel('Wage')
pyplot.title('Details')

# Print the chart
pyplot.show()


# In[ ]:




