import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def transform_age(age):
    if isinstance(age, str):
        return 0
    if age < 20:
        return 1
    elif age < 30:
        return 2
    elif age < 40:
        return 3
    elif age < 50:
        return 4
    elif age < 60:
        return 5
    else:
        return 6

def transform_categorical(dict_, x):
    if x not in dict_.keys():
        return 0
    return dict_[x]

def transform_membership(month):
    if isinstance(month, str):
        return 0
    if month < 24:
        return 1
    elif month < 60:
        return 2
    elif month < 120:
        return 3
    else: 
        return 4

def transform_income(income):
    if isinstance(income, str):
        return 0
    if income < 5e4:
        return 1
    elif income < 1e5:
        return 2
    elif income < 5e5:
        return 3
    else:
        return 4

def transform_value(df):
    for col in tqdm(categorical_col):
        df[col] = df[col].apply(lambda x: transform_categorical(list_encoded[col], x))
    df['age'] = df['age'].apply(lambda x: transform_age(x))
    df['antiguedad'] = df['antiguedad'].apply(lambda x: transform_membership(x))
    df['renta'] = df['renta'].apply(lambda x: transform_income(x)) 
    return df

def extract_information(df, time_stamp='2016-05-28', test=False):
    if not test:
        df = df[df['fecha_dato'] == time_stamp]
        df = df[df['age']!=' NA']
    df.fillna(method='ffill', inplace=True)

    df['antiguedad'] = df['antiguedad'].apply(lambda x: pd.to_numeric(str(x).replace('[^\d.-]', ''), errors='coerce'))
    df = transform_value(df)
    
    personal_information_encoded = {}
    label_per_user = {}
    for _, rcd in tqdm(df.iterrows(), total=len(df)):
        user_id = rcd['ncodpers']
        user_information = rcd[categorical_col + interval_col].tolist()
        personal_information_encoded[user_id] = user_information
        if test:
            continue
        user_label = rcd[label_col].tolist()
        label_per_user[user_id] = user_label
        
    return personal_information_encoded, label_per_user

def get_history_purchase(df, user_id, time_stamp='2016-05-28'):
    item = df.columns[-24:]
    purchase_history = df[(df['fecha_dato'] < time_stamp) & df['ncodpers'].isin(user_id)][item.tolist() + ['ncodpers']]
    purchase_history.fillna(0, inplace=True)
    purchase_history[item] = purchase_history[item].astype('int8')
    
    item_buy_per_user = {}
    for i, rcd in tqdm(purchase_history.iterrows(), total=len(purchase_history)):
        ncodpers = rcd['ncodpers']
        items_bought = np.where(rcd[item] == 1)[0].tolist()
        
        if ncodpers not in item_buy_per_user:
            item_buy_per_user[ncodpers] = []
        
        item_buy_per_user[ncodpers].extend(items_bought)
    return item_buy_per_user

def get_dataset(user_feature, item_bought, label=None):
    X = []
    y = []
    idx = []
    for user_id in tqdm(user_feature.keys()):
        list_bought = item_bought[user_id]
        list_bought = [i+1 for i in list_bought]
        list_bought = list_bought[-20:] if len(list_bought) >= 20 else [0]*(20 - len(list_bought)) + list_bought
        X.append(user_feature[user_id] + list_bought)
        if label is None:
            idx.append(user_id)
            continue
        y.append(label[user_id])
    #X_trainset = [[personal_in4, item_last_buy]*n_user]
    return X, y, idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output-dir', type=str, default='data/')
    parser.add_argument('--train-file', type=str, default='train_ver2.csv')
    parser.add_argument('--test-file', type=str, default='test_ver2.csv')
    args = parser.parse_args()

    print('Loading dataset into DataFrame...')
    df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    categorical_col = ['ind_empleado', 'pais_residencia', 'sexo', 
                       'ind_nuevo', 'tiprel_1mes', 'indresi', 'indext', 
                       'conyuemp', 'canal_entrada', 'indfall', 
                       'nomprov', 'ind_actividad_cliente', 'segmento']
    
    interval_col = ['antiguedad', 'age', 'renta']
    label_col = df.columns[-24:]
    list_encoded = {}

    for col in categorical_col:
        mapping_value = {}
        categorical = list(df[col].unique())
        for id, cate in enumerate(categorical):
            mapping_value[cate] = id+1
        list_encoded[col] = mapping_value

    print('Preprocess & Dump trainset...')
    personal_in4, label = extract_information(df)
    user_id = list(personal_in4.keys())
    item_last_buy = get_history_purchase(df, user_id)
    X, y, _ = get_dataset(personal_in4, item_last_buy, label=label)

    os.mkdir(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir,'X_train.npy'), X)
    np.save(os.path.join(args.output_dir,'y_train.npy'), y)
    
    print('Preprocess & Dump testset...')
    test_personal_information, _ = extract_information(test_df, test=True)
    
    user_id_test = list(test_personal_information.keys())
    
    test_item_buy = get_history_purchase(df, user_id_test, time_stamp='2016-06-28')
    
    X_test, _, X_idx = get_dataset(test_personal_information, test_item_buy)
    
    np.save(os.path.join(args.output_dir,'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir,'test_idx.npy'), X_idx)
    
    assert len(X_test) == len(test_df)