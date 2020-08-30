import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def crossvalid(xx, yy, model, cvf):
    err_trn = []
    err_tes = []
    r_2_tes = []
    r_2_trn = []
    test_point = pd.DataFrame(columns = xx.columns)
    for train_index, test_index in cvf.split(xx):
        x_trn = pd.DataFrame(np.array(xx)[train_index], columns =xx.columns)
        x_tes = pd.DataFrame(np.array(xx)[test_index], columns = xx.columns)
        y_trn = np.array(yy)[train_index]
        y_tes = np.array(yy)[test_index]
        model.fit(x_trn,y_trn)
        x_trn_pred = model.predict(x_trn)
        x_tes_pred = model.predict(x_tes)

        point = pd.concat([x_tes, pd.Series(y_tes), pd.Series(x_tes_pred)], axis = 1)
        test_point = pd.concat([test_point, point])

        err_tes.append(mean_squared_error(x_tes_pred, y_tes))
        err_trn.append(mean_squared_error(x_trn_pred, y_trn))
        r_2_tes.append(r2_score(y_tes, x_tes_pred))
        r_2_trn.append(r2_score(y_trn, x_trn_pred))
    v_tes = np.sqrt(np.array(err_tes))
    v_trn = np.sqrt(np.array(err_trn))
    print ("RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... train" % (v_trn.mean(), v_trn.std(),v_trn.min(), v_trn.max(),np.array(r_2_trn).mean()))
    print ("RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... test" % (v_tes.mean(), v_tes.std(), v_tes.min(), v_tes.max(), np.array(r_2_tes).mean()))
    ret = {}
    ret['trn_mean'] = v_trn.mean()
    ret['trn_std'] = v_trn.std()
    ret['trn_r2'] = np.array(r_2_trn).mean()
    ret['tes_mean'] = v_tes.mean()
    ret['tes_std'] = v_tes.std()
    ret['tes_r2'] = np.array(r_2_tes).mean()
    return ret, v_tes.mean()

def model_run(ret, rmse, model_type, cvf):
    if model_type == "conv":
        print("----- Conventional model -----")
    elif model_type == "prop":
        print("----- Proposed model -----")
    elif model_type == "prop2":
        print("----- Proposed model 2 -----")

    feat, target = data_load(model_type)

    gsearch = gridsearch()
    model = gsearch("RFR")
    ret[f"RFR_{model_type}"], rmse[f"RFR_{model_type}"] = crossvalid(feat, target, model, cvf)
    
    model = gsearch("ETR")
    ret[f"ETR_{model_type}"], rmse[f"ETR_{model_type}"] = crossvalid(feat, target, model, cvf)
    
    model = gsearch("XGB")
    ret[f"XGB_{model_type}"], rmse[f"XGB_{model_type}"] = crossvalid(feat, target, model, cvf)
    return ret, rmse

class gridsearch():
    def __init__(self, rs =1126):
        self.rs = rs

    def __call__(self, model):
        return self.cvmodel(model)

    def best_model_(self, model, feat, target):
        cvmodel = self.cvmodel(model)
        cvmodel.fit(feat, target)
        print(cvmodel.best_params_)
        model = cvmodel.best_estimator_
        return model

    def best_params_(self, model, feat, target):
        cvmodel = self.cvmodel(model)
        cvmodel.fit(feat, target)
        ntrees = cvmodel.best_params_["n_estimators"]
        return ntree

    def cvmodel(self, model):
        if model == "RFR":
            cvmodel = GridSearchCV(RandomForestRegressor(n_jobs = -1, random_state = self.rs),
                                            param_grid = {'n_estimators':[500, 1000]}, cv = 5, n_jobs = -1)
        elif model == "ETR":
            cvmodel = GridSearchCV(ExtraTreesRegressor(n_jobs = -1, random_state = self.rs),
                                                       param_grid = {'n_estimators':[500, 1000]},
                                                       cv = 5,
                                                       n_jobs = -1)   
        elif model == "XGB":
            cvmodel = GridSearchCV(XGBRegressor(n_jobs = -1, importance_type = 'total_gain', rondom_state =self.rs),
                                                param_grid = {'n_estimators':[500, 1000],
                                                              'max_depth':[6,7,8],
                                                              'learning_rate':[0.1,0.05],
                                                              'subsample':[0.8,0.9,1],
                                                              'colsample_bytree':[0.8,0.9,1]},
                                                              cv = 5, 
                                                              n_jobs = -1)
        else:
            warnings.warn("Machine Learning model cannot defined")
        return cvmodel



def posterior(x, p_x, p_y, model):
    if len(p_x.shape) == 1:
        model.fit(p_x.reshape(-1, 1), p_y)
        mu, sigma = model.predict(x.reshape(-1, 1), return_std = True)
    else:
        model.fit(p_x, p_y)
        mu, sigma = model.predict(x, return_std = True)
    ind = np.where(sigma == 0)
    sigma[ind] = 1e-5
    return mu, sigma

def EI(mu, sigma, cur_max):
    Z = (mu - cur_max)/ sigma
    ei = (mu - cur_max) * norm.cdf(Z) + sigma*norm.pdf(Z)
    return ei

def data_load(model_type, option = False, data_opt = False):
    if model_type == "conv":
        data = pd.read_csv("data/conventional.csv").drop("Unnamed: 0", axis = 1)
        feat = data.loc[:, "Mn":"Contact time, s"]
        target = data.loc[:, "Y(C2), %"]
    elif model_type == "prop":
        data = pd.read_csv("data/proposed.csv").drop("Unnamed: 0", axis = 1)
        feat = data.loc[:, "Mn":"Contact time, s"]
        target = data.loc[:, "Y(C2), %"]
    elif model_type == "prop2":
        data = pd.read_csv("data/proposed.csv").drop("Unnamed: 0", axis = 1)
        feat = data.loc[:, "1_AW":"Contact time, s"]
        target = data.loc[:, "Y(C2), %"]
    
    if option:
        idx = (data.loc[:,'Nr of publication'] <= 421)
        feat = feat[idx]
        target = target[idx]

    if data_opt:
        return feat, target, data
    else:
        return feat, target    

def plot_importance(model, labels, topk, save = True, fname = "sample"):
    plt.figure(figsize =(6,6))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    topk_idx = indices[-topk:]
    plt.barh(range(len(topk_idx)), importances[topk_idx], color = 'blue', align = 'center')
    plt.yticks(range(len(topk_idx)), labels[topk_idx])
    plt.ylim([-1, len(topk_idx)])
    plt.xlabel("Feature Importance")
    if save:
        plt.savefig(f"out/{fname}.png", format = "png",  dpi = 1000, bbox_inches ="tight")
    plt.close()

def one_shot_plot(feat, target, model, fname, xylim = [0,35],  random_state =1126):
    plt.figure()
    plt.subplot().set_aspect('equal')
    x_train, x_test, y_train, y_test = train_test_split(feat, target, test_size= 0.1, random_state = random_state)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    plt.plot(y_test, y_test_pred, 'o', c = 'red', markersize = 3,alpha = 0.4, label = 'test')
    plt.plot(y_train, y_train_pred, 'o', c = 'blue', markersize = 3, alpha= 0.4,label = 'train')

    plt.plot([-100,200],[-100,200], c='0', ls ='-', lw = 1.0)
    plt.xlim(xylim)
    plt.ylim(xylim)
    plt.xlabel("Experimental Yield [%]")
    plt.ylabel("Predicted Yield [%]")
    plt.savefig(f"out/{fname}.png", dpi =600, bbox_inches = "tight")