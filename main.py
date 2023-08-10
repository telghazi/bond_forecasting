import pandas as pd 

# Import data 
dataset = pd.read_csv(path)


model = ALSTM(d_feat = len(dataset.columns)-1, )
model.fit(dataset)
model.predict(dataset)
