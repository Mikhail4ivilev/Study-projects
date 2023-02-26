
import os

import pandas as pd
import numpy as np

import kaggle
import subprocess
import zipfile

class RecommendDataBase():
    
    """
    Separate class that stores all data required for our model.
    
    Parameters:
        _req_prod_cols - list of requierd column names in product dataframe;
        _req_trans_cols - list of requierd column names in transactions dataframe;
        _prod_file_name - name of product dataframe file;
        _trans_file_name - name of transactions dataframe file;
        _req_folders - tuple of required folders is model directory;
        _model_data_names - names of data files that our model requires;
        directory - path of the model directory. By default - current directory.
    
    Methods:
       get_req_folders() - returns parameter _req_folders;
       get_model_data_names() - returns parameter _model_data_names;
       _data_check(data, req_cols, msg) - checks if dataframe is empty and/or contains required columns;
       SP_table_FILL(target, source) - provides merge target table with source table. 
                                       It is assumed that source table stores actual information.
       prepare_model_data() - creates main, train and test data for the model.
       
    """
    _req_prod_cols = [
                     'product_id'
                    ,'product_name'
                    ,'aisle_id'
                    ,'department_id'
                    ,'aisle'
                    ,'department'
            ]
    
    _req_trans_cols = [
                     'order_id'
                    ,'user_id'
                    ,'order_number'
                    ,'order_dow'
                    ,'order_hour_of_day'
                    ,'days_since_prior_order'
                    ,'product_id'
                    ,'add_to_cart_order'
                    ,'reordered'
            ]
 
    _prod_file_name = 'products.csv'
    _trans_file_name = 'transactions.csv'
    
    _req_folders = ('1_input', '2_model', '3_output')   
    _model_data_names = ('main_data.csv', 'train_data.csv', 'test_data.csv')
  
    def __init__(self, directory:str=os.getcwd()):
        
        assert isinstance(directory, str), 'Wrong type of input!'
        
        assert os.path.exists(directory), 'Directory does not exists!'
        
        assert (
            len(set(os.listdir(directory)) & set(self._req_folders)) == 3
            ), 'Directory does not contain required folders!'
        
        self.directory = directory
        
    def get_req_folders(self) -> tuple:
        
        """ Returns value of _req_folders """
        
        return self._req_folders
    
    def get_model_data_names(self) -> tuple:
        
        """ Returns value of _model_data_names """
        
        return self._model_data_names
    
    def _data_check(self, data:pd.DataFrame, req_cols:list, msg:str) -> bool:
        
        """ 
        Simple check of dataframe: returns True if dataframe is not empty 
        and dataframe сontains required columns
        
        """
        out = True
        if not (len(data) > 0): # is empty
            msg += ' is empty!'
            out = False
        
        if len(set(req_cols) - set(data.columns)) > 0: # has wrong columns
            msg += ' does not contain required columns!'
            out = False
        
        if not out:
            print(msg)
            
        return out
        
    def SP_table_FILL(self, target:str, source:str) -> None:
        
        """ Provides update, insert and delete actions on table (target) using another table (source) """
        
        assert isinstance(target, str) * isinstance(source, str), 'Wrong type of input!'
        
        assert (
             target in (self._prod_file_name, self._trans_file_name)
            ,f'Target most one of ({self._prod_file_name}, {self._trans_file_name})!'
            )
        
        assert os.path.exists(source), f'Path {source} does not exists!'
        
        src = pd.read_csv(source)
            
        if target == self._prod_file_name:
            assert self._data_check(data=src, req_cols=self._req_prod_cols, msg=source)
        
        else:
            assert self._data_check(data=src, req_cols=self._req_trans_cols, msg=source)
            
        trgt = pd.read_csv(os.path.join(self._req_folders[0], target))
        
#         Hash_key to merge tables and output columns
            
        for data in (trgt, src):
            if target == self._prod_file_name:     
                data['hash_key'] = data['product_id'].apply(lambda x: hash(str(x)))
                out_cols = self._req_prod_cols
            
            else:
                data['hash_key'] = data[['order_id', 'user_id', 'product_id']]                                                        .apply(lambda row: hash(''.join(row.astype(str))), axis=1)
                out_cols = self._req_trans_cols
            
#          Update
            
        df_upd = src['hash_key'].to_frame().merge(trgt, on='hash_key', how='inner')[out_cols]  
            
#          Insert
            
        df_ins = src.assign(dummy=1)[['dummy', 'hash_key']].merge(trgt, on='hash_key', how='right')
        df_ins = df_ins[pd.isna(df_ins['dummy'])][out_cols]
            
#          Delete
            
        df_del = src['hash_key'].to_frame().merge(trgt.assign(dummy=1), on='hash_key', how='left')
        df_del = df_del[pd.isna(df_del['dummy'])][out_cols]
        

#         Result
            
        df_res = pd.concat([df_upd, df_ins], ignore_index=True)
        df_res.to_csv(os.path.join(self._req_folders[0], target), index=False)
            
        print(f'Rows updated: {df_upd.shape[0]}')
        print(f'Rows inserted: {df_ins.shape[0]}')
        print(f'Rows deleted: {df_del.shape[0]}')
               
    def prepare_model_data(self) -> None:
        
        """ 
        Creates main, train and test data files based on current information of products and transactions
        
        """
        msg = 'Model data preparation'
        
	subprocess.run(['kaggle', 'competitions', 'download', '-c', 'skillbox-recommender-system'])
	with zipfile.ZipFile('skillbox-recommender-system.zip', 'r') as zip_ref:
    		zip_ref.extractall(self._req_folders[0])
	
	os.remove('skillbox-recommender-system.zip')

        prod_file_path = os.path.join(self._req_folders[0], self._prod_file_name)
        trans_file_path = os.path.join(self._req_folders[0], self._trans_file_name)
        
        products = pd.read_csv(prod_file_path)
        trans = pd.read_csv(trans_file_path)
        
        check = (
              self._data_check(products, self._req_prod_cols, 'Products data')
            * self._data_check(trans, self._req_trans_cols, 'Transactions data')
        )
        
        if check:
            df = trans.merge(
                    products[['product_id', 'aisle_id', 'department_id']]
                   ,on='product_id'
                )
        
            df = df.merge(
                 df.groupby('user_id')['order_number'].max().reset_index().rename(columns={'order_number': 'last_order'})
                ,on='user_id'
                )

            df_train, df_test = df[~df['order_number'].eq(df['last_order'])], df[df['order_number'].eq(df['last_order'])]

            for data, file_name in zip((df, df_train, df_test), self._model_data_names):
                model_path = os.path.join(self._req_folders[1], file_name)
                data.to_csv(model_path, index=False)
            
            msg += ' succeed!'
            
        else:
            msg += ' failed!'
        
        print(msg)

