import pandas as pd

df = pd.read_csv('Dummy.csv')
df

import turicreate as tc
tc.visualization.set_target(target='browser')

sf = tc.SFrame.read_csv('dummy.csv')
tc.show(sf['x'], sf['f(x)'])

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns


# Plot the responses for different events and regions
sns.lineplot(x="x", y="f(x)", data=df)


import os
import torch
import numpy as np
import plotly

def target_func(ind):
    result = []
    for x in ind:
        result.append(np.exp(-(x[0]-2)**2) + np.exp(-(x[0]-6)**2/10) + 1/(x[0]**2 + 1))
    return torch.tensor(result)


import plotly.graph_objects as go

x = np.linspace(-2, 10, 100)
x_new  = x.reshape((100, -1))

z = target_func(x_new)
z
x_t = [-0.7878788,  -0.4242424, 0.5454545, 1.515152,  2.969697, 4.787879, 8.30303]
y_t = [0.6273949, 0.8664039, 0.9422861, 1.227739, 0.8915558, 0.9055804, 0.6026705  ]
x_upper = []
y_upper = []
y_lower = []
for y in y_t:
    y_lower.append(y*0.5)

for y in y_t:
    y_upper.append(y*2)
x_rev = x_t[::-1]
y_lower = y_lower[::-1]

y_lower

fig = go.Figure()
fig.add_trace(go.Scatter(x = x, y = z, line_color = "#ffa500",  line = dict( width=4, dash='dot'), name = 'original function'))

fig.add_trace(go.Scatter(x = x_t, y = y_t, line_shape='spline', name = ' estimated function', line_color = "#236B8E",mode='lines+markers', line = dict( width=2)))
fig.update_layout(xaxis_title = 'x = brew-style', yaxis_title='f(x) = brew-quality', title = ' The estimation is a Gaussian process')
fig.show()
