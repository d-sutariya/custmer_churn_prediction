

import pandas as pd
from data.data_utils import DataLoader


d = DataLoader(r"..\data\post_deployment\raw\batch_train.csv")
batch_set = d.load_data(drop_customer_id=False)
encd_batch_set = d.preprocess_data()
encd_batch_set = encd_batch_set.reset_index()


import numpy as np
import pandas as pd
import os 
from pathlib import Path
import featuretools as ft

from features.generate_and_transform_features import FeatureGenerater,FeatureTransformer
from data.data_utils import DataLoader
from dotenv import load_dotenv

env_path = Path('.env')
load_dotenv(env_path)

root_dir = Path(os.getenv('ROOT_DIRECTORY'))
feature_name_path = root_dir/'reports'/'feature_dfs'/'featured_final_train.json'
train_set = pd.read_csv(root_dir/'data'/'interim'/'train_set.csv')
transformed_featured_final_train_set = pd.read_csv(root_dir/'data'/'processed'/'transformed_featured_final_train_set.csv')


combined_set = pd.concat([train_set,encd_batch_set])
combined_set = combined_set.drop(columns='index').reset_index(drop=True).reset_index()


with open(feature_name_path,'r') as f:
    feature_defs = ft.load_features(f)

new_es = ft.EntitySet(id="new_entity")

new_es.add_dataframe(
    dataframe=combined_set,
    dataframe_name='final_train',
    index='index'
)

featured_batch = ft.calculate_feature_matrix(
    features=feature_defs,
    entityset=new_es
)

featured_batch = featured_batch.reset_index()


final_featured_batch = featured_batch.reset_index(drop=True).loc[train_set.shape[0]+encd_batch_set['index'],:]


cleaned_featured_batch = FeatureGenerater(combined_set,final_featured_batch).clean_dataframes(final_featured_batch)


customerid = cleaned_featured_batch['customerID']


transformed_batch = FeatureTransformer(cleaned_featured_batch.drop(columns='customerID')).transform()


_,transformed_aligned_batch = transformed_featured_final_train_set.align(transformed_batch,join='inner',axis=1)


transformed_aligned_batch['customerID'] = customerid.reset_index(drop=True)


transformed_aligned_batch.to_csv(root_dir/'data'/'post_deployment'/'processed'/'transformed_aligned_batch.csv')