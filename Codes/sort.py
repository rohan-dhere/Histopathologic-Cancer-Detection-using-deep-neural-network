import shutil as sh #move files
import pandas as pd #read csv
df_train=pd.read_csv("/mnt/d/histopathologic-cancer-detection/train_labels.csv")
for index, row in df_train.iterrows():
    if row['label']==0:
        a=str(row['id'])+'.tif'
        sh.move(a,'0/')
    elif row['label']==1:
        a=str(row['id'])+'.tif'
        sh.move(a,'1/')
        
        