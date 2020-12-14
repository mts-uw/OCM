import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import argparse


excel = pd.read_excel('data/OCM.xlsx')

desc = pd.read_csv('data/Descriptors.csv',skiprows = [0],index_col='symbol').drop(['Unnamed: 0','AN',
                                                                               'name','period',
                                                                               'ionic radius',
                                                                               'covalent radius','group',
                                                                               'VdW radius',
                                                                               'crystal radius',
                                                                               'a x 106 ',
                                                                               'Heat capacity ',
                                                                               'l',
                                                                               'electron affinity ',
                                                                               'VE', 
                                                                               'Surface energy '],axis=1)
desc=desc.fillna(desc.mean())

#componet elements convert
elements = pd.DataFrame(columns = list(desc.index))
for i in range(excel.shape[0]):
    elements.loc[i,'%s'%(excel.loc[i,'Cation 1'])] = excel.loc[i,'Cation 1 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Cation 2'])] = excel.loc[i,'Cation 2 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Cation 3'])] = excel.loc[i,'Cation 3 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Cation 4'])] = excel.loc[i,'Cation 4 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Cation 5'])] = excel.loc[i,'Cation 5 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Cation 6'])] = excel.loc[i,'Cation 6 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Anion 1'])] = excel.loc[i,'Anion 1 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Anion 2'])] = excel.loc[i,'Anion 2 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Support 1'])] = excel.loc[i,'Support 1 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Support 2'])] = excel.loc[i,'Support 2 mol%']
    elements.loc[i,'%s'%(excel.loc[i,'Support 3'])] = excel.loc[i,'Support 3 mol%']

elements = elements.drop('nan', axis = 1)
elements = elements.fillna('0')    

prom = pd.DataFrame()
for i in range(excel.shape[0]):
    prom.loc[i,'Promotor_%s'%(excel.loc[i,'Promotor'])] = 1
prom = prom.drop('Promotor_nan', axis = 1).fillna(0)
prom = prom.fillna(0)

prep = pd.DataFrame()
for i in range(excel.shape[0]):
	prep.loc[i,'%s'%excel.loc[i,'Preparation']] = 1
prep = prep.drop('n.a.', axis = 1)
prep = prep.fillna(0)   

def comp_times_base(comp, base, sort=False, times=True, attention=False):
    count = 0
    for key, rows in comp.iterrows():
        stack = np.vstack((rows, base))
        if times == True:
            time = np.array(base) * np.array(rows)
            stack = np.vstack((rows, time))

        if sort == True:
            stack = pd.DataFrame(stack).sort_values(
                [0], ascending=False, axis=1)

        stack = pd.DataFrame(stack).iloc[1:, :]
        stack = np.array(stack)

        if count == 0:
            if attention:
                res = np.sum(stack, axis=1)
            else:
                res = np.array(stack.T.flatten())

            count += 1
        else:
            if attention:
                res = np.vstack((res, np.sum(stack, axis=1)))
            else:
                res = np.vstack((res, np.array(stack.T.flatten())))

            count += 1
    return res



name = []
for i in range(1, 9):
    #name.append('%i_AN'%i)
    name.append('%i_AW'%i)
    #name.append('%i_group'%i)
    #name.append('%i_period'%i)
    name.append('%i_atomic radius'%i)
    name.append('%i_electronegativity'%i)
    name.append('%i_m. p.'%i)
    name.append('%i_b. p.'%i)
    name.append('%i_delta_fus H'%i)
    name.append('%i_density'%i)
    name.append('%i_ionization enegy'%i)

matrix = pd.concat([excel.loc[:,'Nr of publication'],elements,prom, prep,
                    excel.loc[:,'Temperature, K':]], axis = 1).astype('float')
idx = (matrix.loc[:,'Contact time, s'] <=10) & (matrix.loc[:,'p(CH4)/p(O2)'] <= 10) & (matrix.loc[:,'Y(C2), %'] <= 34.0) & (matrix.loc[:,'Th'] == 0 )
matrix = matrix.loc[idx].drop([ "Th",'p(CH4), bar','p(O2), bar'], axis =1)
matrix.to_csv('data/conventional.csv')



feat = comp_times_base(elements.loc[:,list(desc.index)].astype('float'),desc.astype('float').T, sort = True)
feat = pd.DataFrame(feat)
feat = feat.iloc[:,:64]
feat.columns = name

matrix = pd.concat([excel.loc[:,'Nr of publication'],elements,feat,prom, prep,
                    excel.loc[:,'Temperature, K':]], axis = 1).astype('float')
idx = (matrix.loc[:,'Contact time, s'] <=10) & (matrix.loc[:,'p(CH4)/p(O2)'] <= 10) & (matrix.loc[:,'Y(C2), %'] <= 34.0) & (matrix.loc[:,'Th'] == 0 )
matrix = matrix.loc[idx].drop(["Th",'p(CH4), bar','p(O2), bar'], axis =1)
matrix.to_csv('data/proposed.csv')
