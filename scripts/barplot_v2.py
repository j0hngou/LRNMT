import os
import json
import numpy as np
import pandas as pd

d = {}
filenames = ['t5-small-finetuned-en-to-it-it', 't5-base-finetuned-en-to-it-it', 't5-small-finetuned-en-to-it-lrs-back-it', 't5-base-finetuned-en-to-it-lrs-back-it', '2teachersdistillbacktranslation-en-it']
for i, filename in enumerate(filenames):
  #-----------------key1: model name-----------------#
  fs = filename.split('-')
  if "lrs" in fs:
    model_name = 'DS (' + u'\u2190'+') ' + fs[1]
  elif '2teachersdistillbacktranslation' in fs:
    model_name = 'Dual teachers + back-translation (' + u'\u2190'+')'
  else: 
    model_name = 'Direct fine-tuning ' + fs[1]

  #-----------------key2: bucket size-----------------# 
  zarray = np.zeros(len(os.listdir('/content/all_data/'+filename)))
  idx_dict = {"64": 0, "128": 1, "512": 2}
  for jsfile in os.listdir('/content/all_data/'+filename):
    bucket_size = jsfile.split('-')[1].split('.')[0]
    idx = idx_dict[bucket_size]
    f = open('/content/all_data/'+filename+'/'+jsfile)
    data = json.load(f)
    zarray[idx] = data['eval_bleu']
    f.close
  d[model_name] = zarray.tolist()

#-----------------plotting-----------------#
d_df = pd.DataFrame(d)
d_df.index = ["<64", "<128", "<512"]
color_list = ['#e6f9e1', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
ax = d_df.plot.bar(rot=0, figsize=(12, 8), color=color_list)
ax.tick_params(axis='both', labelsize=16)
ax.legend(fontsize=14)
ax.set_xlabel("Bucket size",  fontsize=18)
ax.set_ylabel("BLEU score",  fontsize=18)