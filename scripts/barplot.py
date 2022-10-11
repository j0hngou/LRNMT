import os
import json
import numpy as np
import pandas as pd

d = {}
filenames = ['t5-base_ro-finetuned-en-to-it-it', 't5-base_fr-finetuned-en-to-it-it', 't5-base-finetuned-en-to-it-it', 
             't5-base-finetuned-en-to-it-hrs-it', 't5-base-finetuned-en-to-it-lrs-it']
for i, filename in enumerate(filenames):
  #-----------------key1: model name-----------------#
  fs = filename.split('-')
  if len(filename.split('_')) == 2 : 
    model_name = fs[3] + '_' + fs[1].split("_")[1] + '_' + fs[5]
  else:
    if len(fs) == 8:
      model_name = fs[3] + '_' + fs[5] + '_' + fs[6]
    else: model_name = fs[3] + '_' + fs[5]

  #-----------------key2: bucket size-----------------# 
  zarray = np.zeros(len(os.listdir('/content/t5-base/'+filename))) 
  for jsfile in os.listdir('/content/t5-base/'+filename):
    bucket_size = jsfile.split('-')[1].split('.')[0]
    idx = int(np.log2(float(bucket_size))-5) #int(np.log2(float(bucket_size))-4)
    f = open('/content/t5-base/'+filename+'/'+jsfile)
    data = json.load(f)
    zarray[idx] = data['eval_bleu']
    f.close
  d[model_name] = zarray.tolist()

#-----------------plotting-----------------#
d_df = pd.DataFrame(d)
d_df.index = ["<32", "<64", "<128", "<256", "<512"]
ax = d_df.plot.bar(rot=0, figsize=(12,9))
ax.set_xlabel("Bucket size")
ax.set_ylabel("Blue score")