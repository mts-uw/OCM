from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt.learning import ExtraTreesRegressor as opt_ETR
import random
random.seed(1126)
np.random.seed(1126)


plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 2
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'black'


excel = pd.read_excel("data/OCM.xlsx")
elements = pd.DataFrame()
for i in range(excel.shape[0]):
    elements.loc[i, '%s' % (excel.loc[i, 'Cation 1'])
                 ] = excel.loc[i, 'Cation 1 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Cation 2'])
                 ] = excel.loc[i, 'Cation 2 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Cation 3'])
                 ] = excel.loc[i, 'Cation 3 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Cation 4'])
                 ] = excel.loc[i, 'Cation 4 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Cation 5'])
                 ] = excel.loc[i, 'Cation 5 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Cation 6'])
                 ] = excel.loc[i, 'Cation 6 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Anion 1'])
                 ] = excel.loc[i, 'Anion 1 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Anion 2'])
                 ] = excel.loc[i, 'Anion 2 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Support 1'])
                 ] = excel.loc[i, 'Support 1 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Support 2'])
                 ] = excel.loc[i, 'Support 2 mol%']
    elements.loc[i, '%s' % (excel.loc[i, 'Support 3'])
                 ] = excel.loc[i, 'Support 3 mol%']


elements = elements.drop('nan', axis=1)
elements = elements.fillna('0')
first_element = elements.columns[0]
last_element = elements.columns[-1]

prom = pd.DataFrame()
for i in range(excel.shape[0]):
    prom.loc[i, 'Promotor_%s' % (excel.loc[i, 'Promotor'])] = 1

prom = prom.drop('Promotor_nan', axis=1).fillna(0)
prom = prom.fillna(0)

prep = pd.DataFrame()

for i in range(excel.shape[0]):
    prep.loc[i, '%s' % excel.loc[i, 'Preparation']] = 1
prep = prep.drop('n.a.', axis=1)
prep = prep.fillna(0)

matrix = pd.concat([excel.loc[:, 'Nr of publication'], elements, prom, prep,
                    excel.loc[:, 'Temperature, K':]], axis=1).astype('float')

data = matrix
data_ = data.loc[(data.loc[:, 'Nr of publication'] <= 421), :]

fig = plt.figure(figsize=(16, 18))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)

edge = np.arange(650, 1300, 20)
ax1.hist(data.loc[:, 'Temperature, K'], bins=edge,
         ec='black', color='orange', label='2010—2019')
ax1.hist(data_.loc[:, 'Temperature, K'], bins=edge, ec='black', label='—2009')
ax1.set_xlabel('Temperature [K]')
ax1.set_ylabel('Datapoints')
ax1.legend()

edge = np.arange(0, 10, 0.5)
ax2.hist(data.loc[:, 'p(CH4)/p(O2)'], bins=edge,
         ec='black', color='orange', label='2010—2019')
ax2.hist(data_.loc[:, 'p(CH4)/p(O2)'], bins=edge, ec='black', label='—2009')
ax2.set_xlim([-1, 11])
ax2.set_ylim([0, 1300])
ax2.set_xlabel('$P_{CH4}$/$P_{O2}$')
ax2.set_ylabel('Datapoints')
ax2.legend()

edge = np.arange(-5, 3, 0.25)
ax3.hist(np.log10(data.loc[:, 'Contact time, s']),
         bins=edge, ec='black', color='orange', label='2010—2019')
ax3.hist(np.log10(data_.loc[:, 'Contact time, s']),
         bins=edge, ec='black', label='—2009')
ax3.set_xlabel('log(Contact time [s])')
ax3.set_ylabel('Datapoints')
ax3.legend()

bar = (data.loc[:, first_element: last_element] > 0).sum()
bar.to_csv('out/elements.csv')
bar_ = (data_.loc[:, 'Mn':'I'] > 0).sum()
topk = 25
labels = bar.index
indices = np.argsort(bar)[::-1]
topk_idx = indices[:topk]
ax4.bar(range(len(topk_idx)), bar[topk_idx], color='orange',
        align='center', ec='black', label='2010—2019')
ax4.bar(range(len(topk_idx)),
        bar_.loc[labels[topk_idx]], ec='black', label='—2009')
ax4.set_xticks(range(len(topk_idx)))
ax4.set_xticklabels(labels[topk_idx], rotation="vertical")
ax4.legend()
ax4.set_ylabel('Datapoints')
ax4.set_xlim([-1, len(topk_idx)])
ax4.set_ylim([0, 1600])

bar = (data.loc[:, first_element: last_element] > 0).sum(axis=1)
bar_ = (data_.loc[:, "Mn":'I'] > 0).sum(axis=1)
labels = np.arange(1, 9)
count = {}
for i in range(1, 9):
    count[i] = (bar == i).sum()

ax5.bar(range(1, 9), count.values(), color='orange',
        ec='black', label='2010—2019')
count = {}
for i in range(1, 9):
    count[i] = (bar_ == i).sum()
ax5.bar(range(1, 9), count.values(), color='C0', ec='black', label='—2009')
ax5.set_xticks(range(1, 9))
ax5.set_xticklabels(range(1, 9))
ax5.set_ylabel('Datapoints')
ax5.set_xlabel('number of component elements')
ax5.set_xlim([0, 9])
ax5.legend()

plt.savefig("out/Figure_1.svg", format="svg", dpi=400)
plt.close()

plt.figure(figsize=(6, 6))
bar = (data.loc[:, 'Impregnation':'Pechini method'] > 0).sum()
bar_ = (data_.loc[:, 'Impregnation':'Pechini method'] > 0).sum()
topk = 25
labels = bar.index
indices = np.argsort(bar)
topk_idx = indices[:topk]
plt.barh(range(len(topk_idx)), bar[topk_idx], color='orange',
         align='center', ec='black', label='2010—2019')
plt.barh(range(len(topk_idx)),
         bar_.loc[labels[topk_idx]], ec='black', label='—2009')
plt.yticks(range(len(topk_idx)), labels[topk_idx])
plt.legend(loc='lower right')
plt.xlabel('Datapoints')
plt.savefig('out/Figure_1_sub.svg', dpi=400, bbox_inches='tight')
plt.close()

plt.figure(figsize=(28, 10))
plt.rcParams["font.size"] = 18
bar = (data.loc[:, first_element:last_element] > 0).sum()
bar.to_csv('out/elements.csv')
bar_ = (data_.loc[:, first_element:last_element] > 0).sum()
topk = 66
labels = bar.index
indices = np.argsort(bar)[::-1]
topk_idx = indices[:topk]
plt.bar(range(len(topk_idx)), bar[topk_idx], color='orange',
        align='center', ec='black', label='2010 ~ 2019')
plt.bar(range(len(topk_idx)),
        bar_.loc[labels[topk_idx]], ec='black', label='~ 2009')
# plt.xticks(range(len(topk_idx)))
plt.xticks(range(len(topk_idx)), labels[topk_idx])
plt.legend()
plt.ylabel('number of data')
plt.xlim([-1, len(topk_idx)])
plt.ylim([0, 1500])
for x, y in zip(range(len(topk_idx)), bar[topk_idx]):
    plt.text(x, y, y, ha="center", va="bottom")
plt.savefig("out/Figure_S3.svg", format="svg",  dpi=1000)
plt.close()

fig, ax_main = plt.subplots(figsize=(12, 12))
divider = make_axes_locatable(ax_main)
ax_main.set_xlim([0, 105])
ax_main.set_ylim([0, 105])

pad = 0.05

idx = (data.loc[:, "W"] > 0) & (data.loc[:, "Mn"] > 0)
color = "blue"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="Mn/W")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)

ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = (data.loc[:, "Li"] > 0) & (data.loc[:, "Mg"] > 0)
color = "orange"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="Li/Mg")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = (data.loc[:, "La"] > 0)
color = "green"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="La")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = ((data.loc[:, 'W'] == 0) & (data.loc[:, 'Mn'] == 0)) & (
    (data.loc[:, 'Li'] == 0) & (data.loc[:, 'Mg'] == 0)) & ((data.loc[:, 'La'] == 0))
color = "red"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="others")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

ax_main.legend()
fig.savefig("out/Figure_2_2.svg", format="svg", dpi=1200)
plt.close()

fig, ax_main = plt.subplots(figsize=(12, 12))
divider = make_axes_locatable(ax_main)
ax_main.set_xlim([0, 105])
ax_main.set_ylim([0, 105])

pad = 0.05

idx = data.loc[:, "Temperature, K"] < 700 + 273.15
color = "blue"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="T < 700 ℃")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)

ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = (data.loc[:, "Temperature, K"] >= 700 +
       273.15) & (data.loc[:, "Temperature, K"] < 800 + 273.15)
color = "orange"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="700 <= t < 800")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = (data.loc[:, "Temperature, K"] >= 800 +
       273.15) & (data.loc[:, "Temperature, K"] < 900 + 273.15)
color = "green"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="800 <= t < 900")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = data.loc[:, "Temperature, K"] > 900 + 273.15
color = "red"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="900 =< t")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

ax_main.legend()
fig.savefig("out/Figure_2_3.svg", format="svg", dpi=1200)
plt.close()

fig, ax_main = plt.subplots(figsize=(12, 12))
divider = make_axes_locatable(ax_main)
ax_main.set_xlim([0, 105])
ax_main.set_ylim([0, 105])

pad = 0.05

idx = data.loc[:, "p(CH4)/p(O2)"] <= 2
color = "blue"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="p < 2")
ax_main.set_xlabel("")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)

ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = (data.loc[:, "p(CH4)/p(O2)"] > 2) & (data.loc[:, "p(CH4)/p(O2)"] <= 3)
color = "orange"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="2 <= p < 3")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = (data.loc[:, "p(CH4)/p(O2)"] > 3) & (data.loc[:, "p(CH4)/p(O2)"] <= 4)
color = "green"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="3 <= p < 4")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

idx = data.loc[:, "p(CH4)/p(O2)"] > 4
color = "red"
xx, yy, zz = data.loc[idx, "X(CH4), %"], data.loc[idx,
                                                  "S(C2), %"], data.loc[idx, "Y(C2), %"]
ax_main.scatter(xx, yy, s=10, alpha=0.8, color=color, label="4 =< p")

ax_x = divider.append_axes("top", 0.5, pad=pad, sharex=ax_main)
ax_y = divider.append_axes("right", 0.5, pad=pad, sharey=ax_main)
ax_x.xaxis.set_tick_params(labelbottom=False)
ax_y.yaxis.set_tick_params(labelleft=False)
ax_x.hist(xx, bins=100, color=color)
ax_y.hist(yy, bins=100, orientation='horizontal', color=color)

ax_main.legend()
fig.savefig("out/Figure_2_4.svg", format="svg", dpi=1200)
plt.close()
