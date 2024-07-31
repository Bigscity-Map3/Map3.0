import pandas as pd

model_list=['SkipGram','POI2Vec','Teaser','Tale','Hier','CTLE','CACSR']
dataset=['nyc','foursquare_tky','chicago','porto','singapore','sanfransico']
metrics=['loc_clf_acc','loc_clf_f1_macro','loc_pre_acc1','loc_pre_acc5','traj_clf_acc','traj_clf_f1_macro']

d1=[]
d2=[]
res={'dataset':[],'metrics':[],'SkipGram':[],'POI2Vec':[],'Teaser':[],'Tale':[],'Hier':[],'CTLE':[],'CACSR':[]}


for model in model_list:
    for d in dataset:
        file=f"/home/zhangwt/remote/zwt/Map3.0/veccity/cache/{model}/evaluate_cache/{model}_evaluate_{model}_{d}_128.csv"
        row=pd.read_csv(file)
        for m in metrics:
            if model == "SkipGram":
                res['dataset'].append(d)
                res['metrics'].append(m)
            if 'loc_pre' in m:
                res[model].append(row.iloc[0][m]/100)
            else:
                res[model].append(row.iloc[0][m])

df=pd.DataFrame(res)
print(df)
df.to_csv('poi_result.csv')


