# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:01:52 2021

@author: JQS
"""
import os, re
from collections import defaultdict
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import Counter
import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = ['Times New Roman']

from sklearn.metrics.pairwise import pairwise_distances
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import  pytorch_cos_sim
model = SentenceTransformer('paraphrase-mpnet-base-v2')


lakes = pd.read_excel('refs_keywords.xlsx')
keywords = pd.read_excel('keywords_post.xlsx')
keywords_dict = defaultdict(str)
keywords_class = defaultdict(str)
keywords_dict.update(dict(zip(keywords['keyword'], keywords['postprocessing_kwd'])))
keywords_class.update(dict(zip(keywords['keyword'], keywords['class_id'])))

'''
match the keyword with published year
'''
DE_PY=[]
DEscPY=[]
for i, de in enumerate(lakes['DE_new']):
    if isinstance(de,str):
        des = de.split('; ')
        key = [keywords_dict[k] for k in des]
        cla = [keywords_class[k] for k in des]
        new = list(zip(key, [lakes['PY'][i]]*len(key), cla ))
        DE_PY += new
        if i%1000==0:
            print(i)
keywords_PY = pd.DataFrame(DE_PY, columns= ['DE','PY','CL'])
keywords_PY.to_pickle('keywords_PY.pckl')
keywords_PY = keywords_PY.replace('dissolve organic carbon','carbon')

keywords_PY = pd.read_pickle('keywords_PY.pckl')
keywords_PY = keywords_PY[keywords_PY['PY'] < 2021]
year_count = pd.DataFrame.from_dict(Counter(keywords_PY['PY']), orient='index', columns=['count'])
year_count = year_count[year_count.index >= 1900].sort_index()
year_count['PY'] = year_count.index


#====================2.1 Number of papers=====================
aricles_PY = pd.DataFrame.from_dict(Counter(lakes['PY']), orient='index',columns=['count'])
aricles_PY['PY'] = aricles_PY.index
aricles_PY= aricles_PY.loc[(aricles_PY['PY']>=1900)&(aricles_PY['PY']<=2020)].sort_values('PY',ascending=True)
    
fig,ax = plt.subplots(figsize=(6,4))
ax = sns.barplot(y="count", x="PY", data=aricles_PY, palette=sns.color_palette("Blues",n_colors=len(aricles_PY)))
ax.set_xticks(list(range(0,121,10)))
ax.set_xticklabels(range(1900,2021,10))
ax.set_xlabel('Year',fontsize=14)
ax.set_ylabel('Number of annual articles',fontsize=14)

aricles_PY1 = aricles_PY[aricles_PY['PY']<=1970]
ax1 = fig.add_axes([0.18,0.24,0.45,0.25])
ax1.bar(range(len(aricles_PY1)), aricles_PY1['count'], width=0.8, )
ax1.set_xticks(list(range(0,71,10)))
ax1.set_xticklabels(range(1900,1971,10))
ax1.set_title('Zooming in 1900-1970')

fig.tight_layout()
fig.savefig('figures\\2.1-Number of papers.jpg', dpi=400)


#====================3.1-1 cumulative counts of top keywords ==================
df = pd.read_excel('comparison.xlsx')
N=1000
fig,ax = plt.subplots(figsize=(5,4))
ax.fill_between(df.index[:N],0,df['count'][:N],facecolor='#2ec7c9',label='Unprocessed keywords',alpha=0.8)
ax.fill_between(df.index[:N],df['count'][:N],df['Rcount'][:N],facecolor='#b6a2de',label='Rule-baed preprocessing',alpha=0.8)
ax.fill_between(df.index[:N],df['Rcount'][:N],df['Scount'][:N],facecolor='#5ab1ef',label='Semantic-based preprocessing',alpha=0.8)
ax.fill_between(df.index[:N],df['Scount'][:N],df['Dcount'][:N],facecolor='#ffb980',label='Deep clusering',alpha=0.8)
ax.fill_between(df.index[:N],df['Dcount'][:N],df['Pcount'][:N],facecolor='#d87a80',label='Postprocessing',alpha=0.8)
ax.vlines(x=100, ymin=0, ymax=df['Pcount'][100], linestyles='dashed',lw=1)
ax.vlines(x=500, ymin=0, ymax=df['Pcount'][500], linestyles='dashed',lw=1)
ax.vlines(x=1000, ymin=0, ymax=df['Pcount'][1000], linestyles='dashed',lw=1)

ax.set_xlabel('Number of top keywords')
ax.set_ylabel('Total counts of top keywords')
ax.set_xlim(left=0)
ax.set_ylim(0,top=600000)
ax.legend(loc=2, fontsize=8 )
fig.tight_layout()
fig.savefig('figures\\3.1-1-Cumulative counts.jpg', dpi=400)


#====================3.1-2(d) cumulative  number of different keywords==================
keywords_df = pd.read_excel('keyword_list.xlsx')
keywords_df['keyword'] = keywords_df['keyword'].map(str)
keywords_df['preprocessed_kwd'] = keywords_df['preprocessed_kwd'].map(str)

diff_keyword = pd.DataFrame()
for i in keywords_df.index:
    ratio = fuzz.ratio(keywords_df['keyword'][i], keywords_df['preprocessed_kwd'][i])
    if ratio < 75 and isinstance( keywords_df['preprocessed_kwd'][i],str):
        diff_keyword = diff_keyword.append(keywords_df.loc[i,:])
        if i%1000==0:print(i)
        
embeddings1 = model.encode(diff_keyword['keyword'].values)
embeddings2 = model.encode(diff_keyword['preprocessed_kwd'].values)
distances = []
for em1, em2 in zip(embeddings1, embeddings2):
    dis = pytorch_cos_sim(em1, em2)
    distances.append(1-dis.numpy()[0][0])
diff_keyword['dist'] = distances
diff_keyword['dist'] = diff_keyword['dist'].clip(0,1)
with open('diff_keyword.pckl', 'wb') as wf:
    pickle.dump(diff_keyword, wf)
    
    
diff_keyword = pd.read_excel('diff_keyword.xlsx')
with open('diff_keyword.pckl', 'rb') as wf:
    diff_keyword = pickle.load(wf)
dist_quantiles =[]
for i in np.linspace(0, 1, 50):
    dist_quantiles.append(diff_keyword['dist'].quantile(i))
    
fig,ax = plt.subplots(figsize=(3,3))
ax.plot(dist_quantiles,np.linspace(1,0, 50)*len(diff_keyword),c='r')
for x in [0.4]:
    ax.vlines(diff_keyword['dist'].quantile(1-x), 0,x*len(diff_keyword),  colors='b', 
              linestyles='dashed',linewidths=1, alpha=0.7)
    ax.hlines(x*len(diff_keyword),diff_keyword['dist'].quantile(1-x),1, colors='b', 
              linestyles='dashed',linewidths=1, alpha=0.7)
ax.set_xlim(-0.05,1)
ax.set_ylim(0,40000)
plt.gca().invert_xaxis()
ax.set_xlabel('Cosine distance')
ax.set_ylabel('Cumulative number of different forms')
fig.tight_layout()
fig.savefig('figures\\3.1-2(d)-Cumulative number of different forms.jpg', dpi=400)


#====================3.2 optimization of multiple transformer framework and similarity measurement  ===============
'''
Looking at the overall distribution of different distance groups. 
If the distance of similarity is smaller, the distance of dissimilarityis larger, this index is the better.
checking relationship between  distances of the same group and  distances of different group.
'''
# 9 transformer frameworks
model_names = ['stsb-roberta-base', 'stsb-distilroberta-base-v2', 'stsb-mpnet-base-v2',
               'paraphrase-TinyBERT-L6-v2','paraphrase-distilroberta-base-v2','paraphrase-mpnet-base-v2',
               'nli-roberta-base-v2','nli-distilroberta-base-v2','nli-mpnet-base-v2']
models =[SentenceTransformer(i) for i in model_names]
# 8 similarity measuremens
metrics = ['euclidean','manhattan','braycurtis', 'canberra', 
           'chebyshev','seuclidean','sqeuclidean','cosine']

# import the test data
test = pd.read_excel('test_keyword.xlsx')
test = test.sort_values('label_true')
test.index = range(len(test))
test_keywords = test['test_kwd']
label_s = list(test.drop_duplicates('label_true').index ) + [len(test)]

#calculate the mean distances of intra-group and inter-group.
dist_metrics = []
for name, model_ in zip(model_names, models):
    em = model_.encode(test_keywords)
    for metric in metrics:
        dist = pairwise_distances(em, metric=metric)
        if metric=='correlation':
            dist = 1- pairwise_distances(em, metric=metric)
        range_ = np.max(dist) - np.min(dist)
        dist = (dist - np.min(dist)) / range_      #normalization for comparison, 
        for s, e in zip(label_s[:-1],label_s[1:]):
            triu = dist[s:e,s:e][np.triu_indices(n=e-s,k=1)]
            inner_95 = np.quantile(triu, 0.5)
            outer = np.append( dist[s:e, 0:s], dist[s:e, e:], axis=1)
            outer_05 = np.quantile(outer, 0.5)
            dist_metrics.append([name, metric, inner_95, outer_05])
        print(name, metric)
dist_metrics_df = pd.DataFrame(dist_metrics,columns=['model','metric','intra_topic','inter_topic'])

#figure: scatters of different similarity distances of intra-group and inter-group.
fig,axes = plt.subplots(figsize=(3.5, 3.5))
plt.rcParams['font.size'] = 8
colors= sns.color_palette('bright',8)[::-1]
sns.scatterplot(x='intra_topic',y='inter_topic',hue='metric', ax=axes, alpha=0.5,
                data=dist_metrics_df[dist_metrics_df['model']=='paraphrase-mpnet-base-v2'], palette=colors)
axes.plot([0,1],[0,1],'k--', alpha=0.5)
axes.legend(loc='lower right', ncol=2, columnspacing=1)
axes.set_xlim(0,1)
axes.set_ylim(0,1)
axes.set_xlabel('Average distance of intra-topic', fontsize=10)
axes.set_ylabel('Average distance of inter-topic', fontsize=10)
axes.text(x=0.02, y=1.02, s='(a)', fontsize=10, transform=axes.transAxes)
fig.subplots_adjust(left=0.15, bottom=0.12, right=0.96, top=0.95)
fig.savefig('figures\\3.2-1 (a)distribution of distances by der different transformer frameworks.jpg', dpi=400)

#figure (b): distribution of distances under different transformer frameworks.
fig,axes = plt.subplots(figsize=(3.5, 3.5))
data = dist_metrics_df[dist_metrics_df['metric']=='cosine']
data.index = data['model']
data = data.drop(['model','metric'],axis=1)
data =data.stack().reset_index().rename(columns={0:'dist'})

sns.boxplot(x='model',y='dist',hue='level_1', ax=axes, linewidth=0.5, showfliers=False, data =data, palette=['#2ec7c9','#d87a80'])
axes.set_ylim(0,1)
axes.set_xlabel('Model structure', fontsize=10)
axes.set_ylabel('distance of intra-topic or inter-topic', fontsize=9)
axes.set_xticklabels(labels=data['model'].unique(), rotation=270)
axes.legend(loc='upper right', ncol=2, columnspacing=1,title=None)
axes.text(x=0.02, y=1.03, s='(b)', fontsize=10, transform=axes.transAxes)

fig.subplots_adjust(left=0.15, bottom=0.5, right=0.96, top=0.95)
fig.savefig('figures\\3.2-1 (b)distribution of distances by der different transformer frameworks.jpg', dpi=400)


#====================3.3 Evolution of Lake-related hot topics ===============
DE_PY=[]
for i, de in enumerate(lakes['DE_new']):
    if isinstance(de,str):
        des = de.split('; ')
        key = [k for k in des]
        new = list(zip(key, [lakes['PY'][i]]*len(key) ))
        DE_PY += new
        if i%1000==0:
            print(i)
keywords_PY1 = pd.DataFrame(DE_PY, columns= ['DE','PY'])
keywords_PY1 =keywords_PY1[keywords_PY1['PY']<2021]

postprocessing = pd.read_excel('keywords_postprocessing.xlsx')
fig,axes = plt.subplots(2,1, figsize=(8,5.2),sharex=True)
plt.rcParams['font.size'] = 9
#(a) 3.3-1(a) Time series of hot topics with NLP-DC.
axes[0].set_prop_cycle((cycler(color=['#c12e34','#e6b600','#0098d9','#2b821d','#005eaa', 
                                 '#801124','#cda819','#32a487','#c2653c','#6b81f2'])))
for i in postprocessing['class_id'][:10]:
    key_word = keywords_PY[keywords_PY['CL']==i]
    key_word = key_word.sort_values('PY')
    stat_PY  = pd.pivot_table(key_word,index='PY',values='CL', aggfunc ='count')
    axes[0].plot(stat_PY.index, stat_PY['CL'],'.-', alpha=0.5, 
            linewidth=1, markersize=3, label=key_word['DE'].iloc[0])
axes[0].set_ylabel('Count of annual topics',fontsize=13)
axes[0].legend(loc='lower left')
axes[0].set_title('Processed by NLP-DC')
axes[0].text(x=0.02, y=1.03, s='(a)', fontsize=10, transform=axes[0].transAxes)

# 3.3-1(b) Time series of the proportion of top topics.
axes[1].set_prop_cycle((cycler(color=['#c12e34','#e6b600','#0098d9','#2b821d','#005eaa', 
                                 '#801124','#cda819','#32a487','#c2653c','#6b81f2'])))
for i in postprocessing['class_id'][:10]:
    key_word = keywords_PY[keywords_PY['CL']==i]
    key_word = key_word.sort_values('PY')
    stat_PY  = pd.pivot_table(key_word,index='PY',values='CL', aggfunc ='count')
    axes[1].plot(stat_PY.index, stat_PY['CL']/year_count.loc[stat_PY.index,'count'],'.-', alpha=0.5, 
            linewidth=1, markersize=3, label=key_word['DE'].iloc[0])
axes[1].set_ylim(0,0.05)
axes[1].set_xticks(range(1900,2021,10))
axes[1].set_xlabel('Year',fontsize=13)
axes[1].set_ylabel('Proportion of annual topics',fontsize=13)
axes[1].legend(loc='lower left')
axes[1].text(x=0.02, y=1.03, s='(b)', fontsize=10, transform=axes[1].transAxes)
fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95)
fig.savefig('figures\\3.2-1(ab) evolution curve-1.jpg', dpi=400)


# 3.3-1 sup. figure: Time series of hot topics with unprocedssed keywords.
fig,axes = plt.subplots( figsize=(6,2.5))
axes.set_prop_cycle((cycler(color=['#c12e34','#e6b600','#0098d9','#2b821d','#005eaa', 
                                 '#801124','#cda819','#32a487','#c2653c','#6b81f2'])))
for i in keywords['keyword'].str.lower()[:10]:
    key_word = keywords_PY1[keywords_PY1['DE']==i]
    key_word = key_word.sort_values('PY')
    stat_PY  = pd.pivot_table(key_word,index='PY',values='DE', aggfunc ='count')
    axes.plot(stat_PY.index, stat_PY['DE']/year_count.loc[stat_PY.index,'count'],'.-', alpha=0.5, 
            linewidth=1, markersize=3, label=key_word['DE'].iloc[0])
axes.set_xlim(1900,2020)
axes.set_xticks(range(1900,2030,10))

#axes.set_ylim(0,0.05)
axes.set_xlabel('Year',fontsize=11)
axes.set_ylabel('Proportion of topics',fontsize=13)
axes.legend(loc='lower left')
fig.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.9)
fig.savefig('figures\\3.2-1sup evolution curve.jpg', dpi=400)


# 3.3-1(c) Number of keywords` counts > 10, proportion > 0.001.
#按多年进行统计
syear, every =1950, 5
keywords_PY2 = keywords_PY.loc[(keywords_PY['PY']>syear) &(keywords_PY['PY']<2021)]
keywords_PY2['year_id'] = keywords_PY2['PY'].apply(lambda x: (x-syear-1)//every)
keywords_PY2 = keywords_PY2[keywords_PY2['DE']!='']
year_count['year_id'] = year_count['PY'].apply(lambda x: (x-syear-1)//every)
years_total = pd.pivot_table(year_count[year_count['PY']>syear],index='year_id',values='count', aggfunc =np.sum)

# abundance
diverse_list =[]
groups = keywords_PY2.groupby('year_id')
for year_id in range(14):
    word_df = groups.get_group(year_id)
    word_period = pd.pivot_table(word_df, index='DE',values='year_id', aggfunc ='count'
                                 ).sort_values('year_id',ascending=False)
    diverse_list.append( list(word_period[word_period['year_id'] >= 10].index))
len_1 = [len(i) for i in diverse_list]

diverse_list =[]
for year_id in range(14):
    word_df = groups.get_group(year_id)
    word_period = pd.pivot_table(word_df, index='DE',values='year_id', aggfunc ='count'
                                 ).sort_values('year_id',ascending=False)  / years_total['count'][year_id]
    diverse_list.append( list(word_period[word_period['year_id'] >= 5/5000].index))
len_2 = [len(i) for i in diverse_list]

DE_PY=[]
for i, de in enumerate(lakes['DE_new']):
    if isinstance(de,str):
        des = de.split('; ')
        key = [k for k in des]
        cla = [keywords_class[k] for k in des]
        new = list(zip(key, [lakes['PY'][i]]*len(key), cla ))
        DE_PY += new
        if i%1000==0:
            print(i)
keywords_PY3 = pd.DataFrame(DE_PY, columns= ['DE','PY','CL'])
keywords_PY3 = keywords_PY3.replace('dissolve organic carbon','carbon')

fig,axes = plt.subplots(1,2, figsize=(8,4))
axes[0].plot(len_1,'^-', c='#001852', markersize=6, label = 'counts > 10')
axes[0].set_xlim(-0.5,11.5)
axes[0].set_xticks(np.linspace(-0.5, 13.5, 8))
axes[0].set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020'])
axes[0].set_xlabel('Year',fontsize=11)
axes[0].set_ylabel('Number of topics',fontsize=13)
axes[0].set_title('Abundance')
axes[0].text(x=0.05, y=1.03, s='(c)', fontsize=10, transform=axes[0].transAxes)
axes[0].spines['left'].set_color('#001852')

ax2 = axes[0].twinx()
ax2.plot(len_2,'d-', c='#e01f54', markersize=6, label='proportion > 1‰')
ax2.set_ylabel('Number of topics',fontsize=13)
ax2.spines['right'].set_color('#e01f54')
ax2.legend()
axes[0].legend()

# 3.3-1(d) Number of keywords within one topic.
cl_kw = {24470:'sediment',25270:'carbon',25106:'phosphorus',26678:'climate change',24471:'fish',26679:'holocene', 
         26923:'phytoplankton',26677:'eutrophication'}
axes[1].set_prop_cycle((cycler(color=['#c12e34','#e6b600','#0098d9','#2b821d','#005eaa', 
                                 '#801124','#cda819','#32a487','#c2653c','#6b81f2'])))

for cl in cl_kw:
    keywords_PY2 = keywords_PY3[keywords_PY3['CL']==cl]
    keywords_PY2 = keywords_PY2.loc[(keywords_PY2['PY']>syear) &(keywords_PY2['PY']<2021)]
    keywords_PY2['year_id'] = keywords_PY2['PY'].apply(lambda x: (x-syear-1)//every)
    keywords_PY2 = keywords_PY2[keywords_PY2['DE']!='']
    
    diverse_list =[]
    groups = keywords_PY2.groupby('year_id')
    for year_id in range(14):
        if year_id not in keywords_PY2['year_id'].unique():
            diverse_list.append([])
        else:
            word_df = groups.get_group(year_id)
            word_period = pd.pivot_table(word_df, index='DE',values='year_id', aggfunc ='count'
                                         ).sort_values('year_id',ascending=False)
            diverse_list.append( list(word_period[word_period['year_id'] >= 5].index))
    len_ = [len(i) for i in diverse_list]
    axes[1].plot(len_,'.-', markersize=6, label=cl_kw[cl])

axes[1].set_xticks(np.linspace(-0.5, 13.5, 8))
axes[1].set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020'])
axes[1].legend()
axes[1].set_xlabel('Year',fontsize=11)
axes[1].set_ylabel('Number of keywords with count > 5',fontsize=11)

axes[1].set_title('Within Topic')
axes[1].text(x=0.05, y=1.03, s='(d)', fontsize=10, transform=axes[1].transAxes)
#fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95)
fig.tight_layout()
fig.savefig('figures\\3.3-1(cd) Time series of hot topics and diversity .jpg', dpi=400)

#====================3.3 Evolution of dependent topics ===============
lakes_ab = pd.read_excel('..\\lakes_ref_info\\lake_ref_list_from_WOS.xlsx')

algaes_kwds = keywords.loc[(keywords['class_id']==26923)|(keywords['class_id']==25107)|(keywords['class_id']==22580)|(keywords['class_id']==25671)|(keywords['class_id']==27070)|(keywords['class_id']==25226),:]
algaes_key = algaes_kwds.sort_values('key_count', ascending=False)['keyword'].str.lower().iloc[:50].unique()
algaes_pat = re.compile(' | '.join(algaes_key))
P_kwds = keywords.loc[(keywords['class_id']==25106)|(keywords['class_id']==25433),:]
P_key = P_kwds.sort_values('key_count', ascending=False)['keyword'].str.lower().iloc[:50].unique()
P_pat = re.compile(' | '.join(P_key))
N_kwds = keywords.loc[(keywords['class_id']==25108)|(keywords['class_id']==25396)|(keywords['class_id']==22755)|(keywords['class_id']==25538)|(keywords['class_id']==26927)|(keywords['class_id']==25567)|(keywords['class_id']==25958),:]
N_key = N_kwds.sort_values('key_count', ascending=False)['keyword'].str.lower().iloc[:50].unique()
N_pat = re.compile(' | '.join(N_key))

lakes['Chla_N'], lakes['Chla_P'] =0, 0
lakes['N_key'], lakes['P_key'] = '', ''
for j in lakes_ab.index:
    abstract = lakes_ab['AB'][j] 
    if isinstance(abstract,str ):
        abst = ' '+abstract.lower() + ' '
        if re.search(algaes_pat, abst):
            k = re.findall(N_pat, abst)
            if k:
                lakes.loc[j,'Chla_N'] = 1
                lakes.loc[j,'N_key'] = str(set(k))
            k = re.findall(P_pat, abst)
            if k:
                lakes.loc[j,'Chla_P'] = 1
                lakes.loc[j,'P_key'] = str(set(k))
        if j%2000==0: print(j)
    else:
        continue

lakes1= lakes.copy()
lakes1['abstract'] = lakes_ab['AB']
lakes2 = lakes1.loc[(lakes1['Chla_N']>0) | (lakes1['Chla_P']>0), :]
lakes2.to_excel('chla_n_p.xlsx', index=False)  #保存

#Sankey plot for overviewing the dependency 
import pyecharts.options as opts
from pyecharts.charts import Sankey

dataN = pd.read_excel('nodes_value.xlsx',sheet_name='N')
dataP = pd.read_excel('nodes_value.xlsx',sheet_name='P')
nodes = [{'name':'Phytoplankton'},{'name':'Only N'},{'name':'N & P'},{'name':'Only P'},{'name':'N'},{'name':'P'}]
nodes += [{'name':i } for i in dataN['name']] + [{'name':i } for i in dataP['name']] 

links=[{"source": "Phytoplankton", "target": "Only N", "value": 1303},
       {"source": "Phytoplankton", "target": "N & P", "value": 2277},
       {"source": "Phytoplankton", "target": "Only P", "value": 2672},
       {"source": "Only N", "target": "N", "value": 1303},
       {"source": "N & P", "target": "N", "value": 2277/2},
       {"source": "N & P", "target": "P", "value": 2277/2},
       {"source": "Only P", "target": "P", "value": 2672},
       ]
links += [{"source": "N", "target": i, "value": j*2441.5/5526} for i,j in zip(dataN['name'], dataN['value'])]+[{"source": "P", "target": i, "value": j*3810.5/8094} for i,j in zip(dataP['name'], dataP['value'])]

(
    Sankey(init_opts=opts.InitOpts(width="900px", height="500px"))
    .add(
        series_name="",
        nodes = nodes,
        links = links,
        itemstyle_opts=opts.ItemStyleOpts(border_width=0.5, border_color="#FFFFFF"),
        label_opts=opts.LabelOpts(position="right",font_family='Times New Roman',font_size=14),
        linestyle_opt=opts.LineStyleOpts(color="target", curve=0.5, opacity=0.4),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title=""))
    .render("figures\\A-2-figure2.2.html")
)

# trend of number of N-Chla and P-chla
chla_n_p = pd.read_excel('chla_n_p.xlsx',sheet_name='Sheet1')
stat_N = pd.pivot_table(chla_n_p,index='PY',values='Chla_N', aggfunc =np.sum).iloc[:-1,:]
stat_P = pd.pivot_table(chla_n_p,index='PY',values='Chla_P', aggfunc =np.sum).iloc[:-1,:]

fig,axes = plt.subplots(1,2, figsize = (8,4))
#fig,ax = plt.subplots()
axes[0].plot(stat_N.index, stat_N['Chla_N'], linewidth=1,c='g',marker='d',label='N-Phytoplankton',alpha=0.6)
axes[0].plot(stat_P.index, stat_P['Chla_P'], linewidth=1,c='m',marker='o',label='P-Phytoplankton',alpha=0.6)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Counts of dependency',fontsize=13)
axes[0].set_xlim(1990,2020)
axes[0].set_ylim(0,400)
axes[0].legend(loc='upper right')
axes[0].text(x=0.03, y=0.96, s='(b)', fontsize=10, transform=axes[0].transAxes)

# trend of proportion of N-Chla and P-chla
def line_regre(x0,y0):
    X = np.array(x0).reshape(-1, 1)
    y = y0.values.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

y1= stat_N['Chla_N']/(stat_N['Chla_N']+stat_P['Chla_P'])
y2= stat_P['Chla_P']/(stat_N['Chla_N']+stat_P['Chla_P'])
axes[1].plot(stat_P.index, y2, 'o',c='m',label='P-Phytoplankton',alpha=0.6)
axes[1].plot(stat_N.index, y1, 'd',c='g',label='N-Phytoplankton',alpha=0.6,)
axes[1].plot([y2.index[0], y2.index[-1]],line_regre(y2.index,y2).predict([[y2.index[0]],[ y2.index[-1]]]), 
             '--',c='m',label ='Trend of P-Phytoplankton')
axes[1].plot([y1.index[0], y1.index[-1]],line_regre(y1.index,y1).predict([[y1.index[0]],[ y1.index[-1]]]), 
             '--',c='g',label ='Trend of N-Phytoplankton')
axes[1].set_xlim(1990,2020)
axes[1].set_xlabel('Year')
axes[1].set_ylim(0.3,0.8)
axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
axes[1].set_ylabel('Proportion',fontsize=13)
axes[1].legend(loc='upper right')
axes[1].text(x=0.03, y=0.96, s='(c)', fontsize=10, transform=axes[1].transAxes)

fig.tight_layout()
fig.savefig('figures\\3.3-2 dependent topics Chla-N-P.jpg', dpi=400)


#====================3.4 Trending up and down and emerging topics ===============
#calculate the proportion of topics every 5 years
syear, every =1970, 5
keywords_PY = pd.read_pickle('keywords_PY.pckl')
data = keywords_PY.loc[(keywords_PY['PY']>syear) &(keywords_PY['PY']<2021)]
data['year_id'] = data['PY'].apply(lambda x: (x-syear-1)//every)
data = data[data['DE']!='']
year_count['year_id'] = year_count['PY'].apply(lambda x: (x-syear-1)//every)
years_total = pd.pivot_table(year_count[year_count['PY']>syear],index='year_id',values='count', aggfunc =np.sum)

word_sort = pd.pivot_table(data,index='DE',values='CL', aggfunc ='count'
                           ).sort_values('CL',ascending=False)
groups = data.groupby('DE')
word_ratio = pd.DataFrame()
for k in word_sort.index[:1001]:
    word_df = groups.get_group(k)
    word_period = pd.pivot_table(word_df, index='year_id',values='DE', aggfunc ='count')
    word_normalized = word_period['DE'] / years_total['count']
    col = k.replace(' ','_')
    word_ratio[col] = word_normalized
word_ratio =word_ratio*1000
word_ratio.to_excel('word_ratio_%d.xlsx'%every)
word_ratio['x'] = word_ratio.index
word_ratio = word_ratio.drop(10)

# fit the temporal trend of topics with genernal linear regression of statsmodels.
import statsmodels.formula.api as smf
results=[]
for col in word_ratio.columns[:1000]:
    my_formula = '%s ~ x' %col
    try:
        lm = smf.glm(formula = my_formula, data=word_ratio).fit()
        result = {'coef_Int':lm.params[0],'coef_time':lm.params[1],
          'p_Int':lm.pvalues[0],'p_time':lm.pvalues[1],
          'conf_Int_low':lm.conf_int().iloc[0,0],'conf_Int_up':lm.conf_int().iloc[0,1], 
          'conf_time_low':lm.conf_int().iloc[1,0],'conf_time_up':lm.conf_int().iloc[1,1], 
          'Number':lm.nobs,'keyword':col }
        results.append(result)
    except:
        continue
trend_three = pd.DataFrame(results)
trend_three.to_excel('trend_three_%d.xlsx'%every)

trend = pd.read_excel('trend_three_%d.xlsx'%every)
# trend = trend[trend['Number']>4]
trend_category = [
trend.loc[(trend['p_Int']>0.05)&(trend['p_time']>0.05)],
trend.loc[(trend['coef_Int']>0)&(trend['p_Int']<=0.05)&(trend['p_time']>0.05)],
trend.loc[(trend['coef_Int']<0)&(trend['p_Int']<=0.05)&(trend['coef_time']>0)&(trend['p_time']<=0.05)],
trend.loc[(trend['p_Int']>0.05)&(trend['coef_time']>0)&(trend['p_time']<=0.05)],
trend.loc[(trend['coef_Int']>0)&(trend['p_Int']<=0.05)&(trend['coef_time']<0)&(trend['p_time']<=0.05)],
trend.loc[(trend['coef_Int']>0)&(trend['p_Int']<=0.05)&(trend['coef_time']>0)&(trend['p_time']<=0.05)],]

#plot 6 types of evolution stage
categs = trend_category[::-1]
labels = {0:'Trending-up', 1: 'Trending-down',2: 'Emerging in 1970',3: 'Emerging after 1970',4: 'Stable',5: 'Fluctuating '}
word_ratio = pd.read_excel('word_ratio_5.xlsx')
#plt.rc('axes', prop_cycle=cycler(color =['#c12e34','#e6b600','#0098d9','#2b821d','#005eaa','#801124','#cda819','#32a487','#c2653c','#89cca8']))

colors = list(plt.rcParams["axes.prop_cycle"])
fig,axes = plt.subplots(3,2,figsize=(6,8),sharex=True)
for i, ax in enumerate(axes.flat):
    for k in range(len(categs[i]))[::-1]:
        w = categs[i].iloc[k]
        ax.plot(range(10),word_ratio[w['keyword']],'.', c=colors[k%10]['color'], alpha=0.5, markeredgewidth=0, )
        ax.plot([0,9], [w['coef_Int'], 9*w['coef_time'] + w['coef_Int']], c=colors[k%10]['color'],alpha=0.6, lw=0.8)
        if k==0:
            ax.plot(range(10),word_ratio[w['keyword']],'.',c='#c12e34', alpha=0.7)
            ax.plot([0,9], [w['coef_Int'], 9*w['coef_time'] + w['coef_Int']],'-', c='#c12e34', linewidth=2)
            ax.annotate(w['keyword'].replace('_',' '), color='#c12e34',
                        xy=(5, 5*w['coef_time'] + w['coef_Int']),
                        xytext=(0.2,0.8), textcoords = 'axes fraction',
                        arrowprops=dict(arrowstyle="->",color='#c12e34',linestyle ='dashed', connectionstyle="arc3,rad=0.3"))
            
    ax.text(0.8,0.95, '(n = %d)'%len(categs[i]), transform=ax.transAxes)
    ax.text(0.03,0.95, '(%s)'%'abcdef'[i], transform=ax.transAxes)
    ax.set_title(labels[i])
    
axes[0,0].set_ylim(0,top=10)
axes[0,1].set_ylim(0,top=25)
axes[1,0].set_ylim(0,top=10)
axes[1,1].set_ylim(0,top=5)
axes[2,0].set_ylim(0,top=10)
axes[2,1].set_ylim(0,top=10)
axes[2,0].set_xticks(np.linspace(-0.5, 9.5, 11))
axes[2,0].set_xticklabels(['1970','1975','1980','1985','1990','1995','2000','2005','2010','2015','2020'], fontsize=8)

axes[2,0].set_xlabel('Year')
axes[2,1].set_xlabel('Year')
axes[0,0].set_ylabel('Topic attention index')
axes[1,0].set_ylabel('Topic attention index')
axes[2,0].set_ylabel('Topic attention index')
fig.tight_layout()
fig.savefig('figures\\3.4-1 Trending up and down and emerging topics.jpg', dpi=400)

# figure: 6 types of evolution stage
for i in range(6):
    fig,ax = plt.subplots(figsize=(3,2.6))
    for k in range(len(categs[i]))[::-1]:
        w = categs[i].iloc[k]
        ax.plot(range(10),word_ratio[w['keyword']],'.', c=colors[k%10]['color'], alpha=0.5, markeredgewidth=0, )
        ax.plot([0,9], [w['coef_Int'], 9*w['coef_time'] + w['coef_Int']], c=colors[k%10]['color'],alpha=0.6, lw=0.8)
        if k==0:
            ax.plot(range(10),word_ratio[w['keyword']],'.',c='#c12e34', alpha=0.7)
            ax.plot([0,9], [w['coef_Int'], 9*w['coef_time'] + w['coef_Int']],'-', c='#c12e34', linewidth=2)
            ax.annotate(w['keyword'].replace('_',' '), color='#c12e34',
                        xy=(5, 5*w['coef_time'] + w['coef_Int']),
                        xytext=(0.2,0.8), textcoords = 'axes fraction',
                        arrowprops=dict(arrowstyle="->",color='#c12e34',linestyle ='dashed', connectionstyle="arc3,rad=0.3"))
            
    ax.text(0.75,0.92, '(n = %d)'%len(categs[i]), transform=ax.transAxes)
    ax.text(0.03,0.92, '(%s)'%'dfcbea'[i], transform=ax.transAxes)
    ytop = 10 if i not in [1,3] else 5 if i !=1 else 25
    ax.set_ylim(bottom=0,top=ytop)
    ax.set_title(labels[i])
    ax.set_xlabel('Year')
    ax.set_ylabel('Proportion of topics ( ‰ )')
    ax.set_xticks(np.linspace(-0.5, 9.5, 6))
    ax.set_xticklabels(['1970','1980','1990','2000','2010','2020'], fontsize=8)
    fig.tight_layout()
    fig.savefig('figures\\3.4-1-part(%s).jpg' %'dfcbea'[i], dpi=400)

#supplement figure:Comparison of slopes of EAP, EIP, TUP
data = pd.read_excel('Fig. Comparison of slopes of EAP, EIP, TUP.xlsx')
fig,ax = plt.subplots(figsize=(3,4))
ax.boxplot(data.T.values.tolist(), patch_artist=True,labels = list(data.columns), 
           boxprops={'color':'k','facecolor':'#0098d9'})
ax.set_ylabel('Slopes of trend')
fig.tight_layout()
fig.savefig('figures\\3.4-1-sup3.jpg', dpi=400)


#supplement figure: Examples of experiencing the two or more evolution patterns
fig,ax = plt.subplots(figsize=(5,4))
for k in [0,1,2,4,5,6,7,8]:
    w = trend_category[0].iloc[k]
    ax.plot(range(10),word_ratio[w['keyword']],'.-', c=colors[k%10]['color'], alpha=0.5, markeredgewidth=0, label=w['keyword'].replace('_',' '))
ax.set_ylim(bottom=0,top=5)
ax.set_xlabel('Year')
ax.set_ylabel('Proportion of topics ( ‰ )')
ax.set_xticks(np.linspace(-0.5, 9.5, 6))
ax.set_xticklabels(['1970','1980','1990','2000','2010','2020'], fontsize=8)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('figures\\3.4-1-sup4.jpg', dpi=400)


#====================3.4 overview for all intercept-slopes. ===============
types=['Intercept~0 & Slope~0',
        'Intercept>0 & Slope~0',
        'Intercept<0 & Slope>0',
        'Intercept~0 & Slope>0',
        'Intercept>0 & Slope<0',
        'Intercept>0 & Slope>0']
marker = ['s', '>', '<', '^', 'd', 'o']
colors = ['#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']

fig,ax = plt.subplots(figsize=(6,5))
ax.vlines(0,-100,100,color='k',linestyles='dashed',zorder=0, alpha=0.3)
ax.hlines(0,-100,100,color='k',linestyles='dashed',zorder=0, alpha=0.3)
#ax.plot([-10,15],[1.4,-2.1],'k-',zorder=0, alpha=0.5) #渐近线：y=-0.14x
for i,c in enumerate(trend_category):
    ax.scatter(c['coef_Int'], c['coef_time'], c=colors[i], marker=marker[i], linewidths=0, label=types[i],alpha=0.6)

#TODO 修改为单点标注
# for i in [3,4,5]:
#     for k in range(3):
#         ax.annotate(trend_category[i]['keyword'].iloc[k], 
#                     xy=(trend_category[i]['coef_Int'].iloc[k], trend_category[i]['coef_time'].iloc[k]),
#                     xytext=(10,10), textcoords = 'offset points',
#                     color=colors[i],arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-0.3"))
ax.set_xlim(-10, 28)
ax.set_ylim(-1.5, 1.2)
ax.set_xlabel('Intercept of GLM for Topics')
ax.set_ylabel('Slope of Topics over time')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig('figures\\A-1-figure3.jpg', dpi=400)

#====================3.4 overview for all intercept-slopes. ===============
import networkx as nx
from nxviz import annotate,highlights, utils
import nxviz as nv

refs = pd.read_excel('refs_keywords_final.xlsx')
de_pairs= []
for k,de in enumerate(refs['DE_final']):
    if isinstance(de,str):
        des = list(set(de.split('; ')))
        if '' in des: des.remove('')
        de_pairs +=[[des[i],des[j],1] if des[i] < des[j] else [des[j],des[i],1]
                    for i in range(len(des)-1) for j in range(i+1,len(des)) ]
        if k%1000==0:print(k)
df = pd.DataFrame(de_pairs,columns=['pairA','pairB','count'])
pairs_df = pd.pivot_table(df,values='count', index=['pairA','pairB'],aggfunc=np.sum)
pairs_df = pairs_df.sort_values('count',ascending=False).reset_index()
#pairs_df.to_excel('co_occurrence_pairs_backup.xlsx')


top1000 = pd.read_excel('nodes_circos.xlsx')
new0 = list(top1000[top1000['category']=='Emerging in 1970']['keyword'])[:20]
new1 = list(top1000[top1000['category']=='Emerging after 1970']['keyword'])[:10]
new = new0 + new1
old = list(top1000[top1000['category']=='Trending-down']['keyword'])[:20]
kw = new + old

pairs_df = pd.read_excel('co_occurrence_pairs_backup.xlsx')
pairs_df['A'] = pairs_df['pairA'].apply(lambda x :1 if x in kw else 0)
pairs_df['B'] = pairs_df['pairB'].apply(lambda x :1 if x in kw else 0)
data = pairs_df.loc[ (pairs_df['A']>0) & (pairs_df['B']>0)]
nodes_class = dict(zip(top1000['keyword'], top1000['category']))
nodes_count = dict(zip(top1000['keyword'], top1000['count']))

# figure:co-occurrence
def circosplot(nx_df, top_edges ):
    G = nx.from_pandas_edgelist(nx_df, 'pairA', 'pairB',['count', 'edge_value'])
    for v in G:
        G.nodes[v]['category'] = nodes_class[v]
        G.nodes[v]['value'] = nodes_count[v]/3600
    nt = utils.node_table(G,)
    pos = nv.layouts.circos(nt,group_by="category", sort_by='value')
    pos_df = pd.DataFrame(pos)
    #center = pos_df.sum(axis=1)
    
    ax = nv.circos(
        G, 
        group_by="category", 
        sort_by='value',
        node_size_by='value',
        node_color_by="category",
        edge_alpha_by="edge_value",
        edge_lw_by = "edge_value",
    )
    
    pos_xmax= pos_df.loc[0].max()
    for node in G.nodes():
        if pos[node][0] > pos_xmax/1.8:
            ax.annotate(text=node, xy=pos[node] + [2,0], ha="left", va="center",rotation=0)
        elif pos[node][0] <- pos_xmax/1.8:
            ax.annotate(text=node, xy=pos[node] - [2,0], ha="right", va="center",rotation=0)
        elif pos[node][1] > pos_xmax/1.8:
            ax.annotate(text=node, xy=pos[node] + [0,1], ha="center", va="bottom",rotation=90)
        elif pos[node][1] < -pos_xmax/1.8:
            ax.annotate(text=node, xy=pos[node] - [0,1], ha="center", va="top",rotation=90)
    #highlights
    for i in range(top_edges):
        u = nx_df['pairA'].iloc[i]
        v = nx_df['pairB'].iloc[i]
        highlights.circos_edge(G, u, v, color="r",lw=1.5, alpha=0.6, group_by="category", sort_by='value')
    return ax

# run the plot function
plt.rcParams['font.size']=8

# (a) top 60
nx_df1  = pd.read_excel('co_occurrence_plot.xlsx',sheet_name='co_occurrence_plot_60')
nx_df1['edge_value'] = nx_df1['count'] / nx_df1['count'].quantile(0.9)
nx_df1['edge_value'] = nx_df1['edge_value'].clip(0, 10*nx_df1['edge_value'].mean())

fig, ax = plt.subplots(figsize=(5,5))
ax = circosplot(nx_df1,top_edges=0)
ax.text(x=0.03, y=0.96, s='(a)', fontsize=10, transform=ax.transAxes)
fig.subplots_adjust(left=0.23, bottom=0.14, right=0.81, top=0.92)
fig.savefig('figures\\3.4-2 (a) co-occurrence of emerging and trending-up.jpg', dpi=400)

# (b) EAP,EIP and TDP
nx_df = data[data['count']>10]
nx_df['edge_value'] = nx_df['count'] / nx_df['count'].quantile(0.9)
nx_df['edge_value'] = nx_df['edge_value'].clip(0, 10*nx_df['edge_value'].mean())

fig, ax = plt.subplots(figsize=(5,5))
ax = circosplot(nx_df,top_edges=0)
ax.text(x=0.03, y=0.96, s='(b)', fontsize=10, transform=ax.transAxes)
fig.subplots_adjust(left=0.12, bottom=0.2, right=0.9, top=0.83)
fig.savefig('figures\\3.4-2 (b) co-occurrence of emerging and trending-up.jpg', dpi=400)

