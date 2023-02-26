
import os

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import euclidean_distances

import asyncio
import nest_asyncio
nest_asyncio.apply()
asyncio.set_event_loop(asyncio.new_event_loop())

from recommend_database import RecommendDataBase

class RecommendModel():
    
    """
    Special class for my recommendation system. Provides train (two steps), predict (two steps) and evaluate methods.
    By default number of recommendations (k) is equal to 10, but it can be changed. If full train is not needed, first 
    step can be skipped because it works for all values of k. If model was trained with value of k greater than required
    for the prediction, there is no need to train model again.
    
    Parameters:
        _modes - tuple with two possible mode values: 'train' and 'inference'. In 'train' mode model uses only part of 
                 data (all purchases but the last one for all users) Submission is not available. In'inference' mode 
                 model uses full data and evaluation is not available;
        _default_k - default number of recommendations in predictions;
        _max_cluster_num - limit number of clusters in elbow method;
        _neigh - NearestNeighbors model from sklearn;
        database - data source for the model;
        mode - 'train' or 'inference';
        _model_data_names - _model_data_names from database;
        _req_folders - _req_folders from database;
        _model_train_data - path to train dataframe;
        _model_eval_data - path to test dataframe;
        _model_emb_path - path to embedding dataframe;
        _model_knn_pt - pattern of the path to k nearest neighbors files;
        _model_sub_path - path to sibmission file
        
    Methods:
        get_default_k() - returns current value of _default_k;
        set_default_k(new) - sets new value of _default_k;
        get_max_cluster_num() - returns current value of _max_cluster_num;
        set_max_cluster_num (new) - sets new value of _max_cluster_num;
        _best_kmeans(x) - provides elbow method and returns best KMeans algorithm for array x;
        _train_step_1() - aggregates data from source, creates embeddings and prepares NearestNeighbors model;
        _train_step_2(k) - calculates k neareset neighbors for each product and saves results into knn file;
        train(full, k) - if full is True or first step has not been done yet, provides both steps of training.
                         In other case it launches only second step;
        _predict_step_1(k) - returns k nearest neighbors for each product. If knn file with k greater than current exists,
                             it uses this file. If not, it launches train method;
        _predict_step_2(k, users) - retrun top k products for each user;
        predict(users, k) - returns top k revelant products for each user. By default _default_k is used;
        _precision(arr1, arr2, n) - returns precision@n score for two arrays;
        evaluate(n) - calculates MAP@n score for the model. Is enabled only with 'train' mode;
        make_submission(n) - createx submission file. Is enabled only with 'inference' mode.
                             
    """    
    _modes = ('train', 'inference')
    _default_k = 10
    _max_cluster_num = 10
    _neigh = None
    
    def __init__(self, database:RecommendDataBase, mode:str='inference'):
        
        assert isinstance(database, RecommendDataBase) * isinstance(mode, str), 'Wrong type of input!'
        assert mode in self._modes, f'Mode must be one of ({self._modes[0]}, {self._modes[1]})'
        
        self.database = database
        self._model_data_names = self.database.get_model_data_names()
        self._req_folders = self.database.get_req_folders()
        
        self.mode = mode
        if self.mode == self._modes[0]:
            self._model_train_data = os.path.join(self._req_folders[1], self._model_data_names[1])
            self._model_eval_data = os.path.join(self._req_folders[1], self._model_data_names[2])
            self._model_emb_path = os.path.join(self._req_folders[1], 'embeddings.csv')
            self._model_knn_pt = os.path.join(self._req_folders[1], '##_nearest_neighbors.csv')
            self._model_sub_path = None
        
        else:
            self._model_train_data = os.path.join(self._req_folders[1], self._model_data_names[0])
            self._model_eval_data = None
            self._model_emb_path = os.path.join(self._req_folders[1], 'embeddings.csv')
            self._model_knn_pt = os.path.join(self._req_folders[1], '##_nearest_neighbors.csv')
            self._model_sub_path = os.path.join(self._req_folders[2], 'submission.csv')

    def get_default_k(self) -> int:
        
        """ Returns current value of _default_k """
        
        return self._default_k
    
    def set_default_k(self, new:int) -> None:
        
        """ Sets new value of _default_k """  
            
        assert isinstance(new, int), 'Wrong type of input!'
        
        self._default_k = new
        
    def get_max_cluster_num(self) -> int:
        
        """ Returns current value of _max_cluster_num """
        
        return self._max_cluster_num
    
    def set_max_cluster_num(self, new:int) -> None:
        
        """ Sets new value of _max_cluster_num """  
            
        assert isinstance(new, int), 'Wrong type of input!'
        
        self._max_cluster_num = new
        
    def _best_kmeans(self, x: np.ndarray) -> KMeans:
        
        """ Returns best KMeans model according to elbow method """
        
        assert isinstance(x, np.ndarray), 'Wrong type of input'
        
        k_limit = self._max_cluster_num
        
        models = {key: None for key in range(1, k_limit + 1)}
        metrics = {key: None for key in range(1, k_limit + 1)}

        for k in range(1, k_limit + 1):
            models[k] = KMeans(n_clusters=k, random_state=17).fit(x)
            centroids, labels = models[k].cluster_centers_, models[k].labels_
    
            metric = 0
            for centroid in range(k):
                metric += euclidean_distances(x[labels==centroid], centroids[centroid, :].reshape(1, -1)).sum(axis=0)[0]
    
            metrics[k] = metric
    
        D = {
            k: abs(metrics.get(k + 1, 0) - metrics[k]) / abs(metrics[k]- metrics.get(k - 1, 0))
            for i in range(2, k_limit + 1)
        }

        best_k = min(D.items(), key=lambda x: x[1])[0]
        
        return models[best_k]
    
    def _train_step_1(self) -> None:
        
        """ Step 1 of training: data aggregation, clustering, embeddings and NearestNeighbors model training """
               
        if not os.path.exists(self._model_train_data):
            self.database.prepare_model_data()
        
        assert os.path.exists(self._model_train_data), f'Path {self._model_train_data} does not exists!'
        
        df = pd.read_csv(self._model_train_data)
        
#         Aggregation

        df_agg = df.groupby(['product_id', 'aisle_id', 'department_id'])           .agg({
                 'user_id': 'nunique'
                ,'order_dow': 'median'
                ,'order_hour_of_day': 'median'
                ,'days_since_prior_order': 'mean'
                ,'add_to_cart_order': 'median'
                ,'reordered': 'sum'
        }).reset_index().rename(
                            columns={
                                 'user_id': 'n_users'
                                ,'order_dow': 'median_dow'
                                ,'order_hour_of_day': 'median_hour_of_day'
                                ,'days_since_prior_order': 'mean_days_between_orders'
                                ,'add_to_cart_order': 'median_cart_position'
                                ,'reordered': 'total_reordered'                                          
                            }
                        )
        df_agg.dropna(inplace=True)
        
        num_cols = ['n_users', 'mean_days_between_orders', 'median_cart_position', 'total_reordered']
        cat_cols = ['aisle_id', 'department_id', 'median_dow', 'median_hour_of_day']
        
        for col in cat_cols:
            df_agg[col] = df_agg[col].astype(np.int32).astype(str)
            
        num_features = df_agg[num_cols]

        cat_features = pd.get_dummies(df_agg[cat_cols])

        cols = num_features.columns.tolist() + cat_features.columns.tolist()

        x = np.hstack([
                 num_features.values.reshape(df_agg.shape[0], -1)
                ,cat_features.values.reshape(df_agg.shape[0], -1)
        ])
        
#         Clustering
        
        best_model = self._best_kmeans(x=x)
        
        df_agg['Cluster'] = best_model.labels_
        
#         Embeddings
        
        prod_emb = df_agg.reset_index().assign(indx=lambda x: x.index)
        prod_emb = prod_emb[['indx', 'product_id'] + num_cols]                                                        .join(pd.get_dummies(prod_emb[cat_cols]))                                                        .join(pd.get_dummies(prod_emb['Cluster']))
        
        prod_emb.to_csv(self._model_emb_path, index=False)
        
#         Nearest neighbors
        
        emb_cols = prod_emb.columns[2:]

        self._neigh = NearestNeighbors(n_jobs=-1, metric='euclidean').fit(prod_emb[emb_cols].values)
        
    def _train_step_2(self, k:int) -> None:
        
        """ Step 2 of training: calculation k nearest neighbors """
        
        prod_emb = pd.read_csv(self._model_emb_path)
        emb_cols = prod_emb.columns[2:]
        
        tmp_res_list = []

        async def knn(prod_id:int) -> None:
    
            """ Coroutine that finds k nearest neighbors to selected product """
    
            neighbors = self._neigh.kneighbors(
                                 prod_emb[prod_emb['product_id'].eq(prod_id)][emb_cols].values
                                ,n_neighbors=k
                            ) 
            tmp = pd.DataFrame({
                         'product_id_x': [prod_id] * k
                        ,'indx': neighbors[1][0]
                        ,'Metric': neighbors[0][0]
            
            })   
            tmp_res_list.append(tmp)
    
        async def main():
    
            """ Main coroutine that schedules the other coroutines """

            tasks = [asyncio.create_task(knn(prod)) for prod in prod_emb['product_id'].values]

            await asyncio.gather(*tasks)
            
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        loop.run_until_complete(loop.shutdown_asyncgens())
    
        prod_knn_df = pd.DataFrame()
        for elem in tmp_res_list:
            prod_knn_df = pd.concat([prod_knn_df, elem], ignore_index=True)

        knn_path = self._model_knn_pt.replace('##', str(k))
        prod_knn_df.to_csv(knn_path, index=False)
            
        asyncio.set_event_loop(asyncio.new_event_loop())    
        
    def train(self, full:bool=True, k:int=None) -> None:
        
        """ Trains model step by step. Provides an option to skip first step """
        
        assert isinstance(full, bool), 'Wrong type of input!'
        
        k = k or self._default_k
        
        assert isinstance(k, int), 'Wrong type of input!'
        
        if full:
            self._neigh = None
        
        if not ((self._neigh is not None) and os.path.exists(self._model_emb_path)):
            self._train_step_1()
            
        self._train_step_2(k=k)
        
        print('Train succeed!')
    
    def _predict_step_1(self, k:int) -> pd.DataFrame:
    
        """ Predict step 1: returns top k neighbors for each product """
        
        knn_files = []
        for file in os.listdir(self._req_folders[1]):
            if file.endswith('_nearest_neighbors.csv'):
                knn_files.append(file)
                
        if len(knn_files) == 0:
            self.train(full=False, k=k)
            
        for knn_file in knn_files:
            cur_k = int(knn_file.split('_')[0])
            if k <= cur_k:
                knn_path = os.path.join(self._req_folders[1], knn_file)
                break
        else:
            self.train(full=False, k=k)
            knn_path = self._model_knn_pt.replace('##', str(k))
        
        out = pd.read_csv(knn_path)
        prod_emb = pd.read_csv(self._model_emb_path)
        
        out = out.merge(prod_emb, on='indx')[['product_id_x', 'product_id', 'Metric']]                                        .rename(columns={'product_id_x': 'product_id', 'product_id': 'product_id_y'})
    
        return out
            
    def _predict_step_2(self, k:int, users:(list, np.ndarray)=None) -> pd.DataFrame:
        
        """ Predict step 2: returns information of top k purchases """
        
        data = pd.read_csv(self._model_train_data)
        
        users = users or np.sort(data['user_id'].unique())
        
        out = data[data['user_id'].isin(users,)].assign(dummy=1)                                                .groupby(['user_id', 'product_id'])                                                .agg({'dummy': 'sum'})                                                .assign(weight=lambda x: x.dummy )                                                .sort_values(['user_id', 'weight'], ascending=False)                                                .reset_index()                                                .groupby('user_id')                                                .apply(lambda x: x.iloc[: k])[['user_id', 'product_id', 'weight']]
    
        return out    
    
    def predict(self, users:(int, list, np.ndarray)=None, k:int=None) -> pd.Series:
        
        """ Return top-k relevant products for each user in array. By default makes prediction for all users """
        
        assert isinstance(users, (int, list, np.ndarray)) + (users is None), 'Wrong type of input!'
                
        k = k or self._default_k
        
        assert isinstance(k, int), 'Wrong type of input!'
        
        prod_neigh = self._predict_step_1(k=k)
        
        if isinstance(users, int):
            users = [users]
        
        user_hist = self._predict_step_2(k=k, users=users)
            
        out = user_hist.merge(prod_neigh, on='product_id')[['user_id', 'product_id', 'weight', 'product_id_y', 'Metric']]
    
        out['Metric'] = (out['Metric'] + 1) / out['weight']
        
        out = out.sort_values(['user_id', 'Metric'], ascending=True)                                                .reset_index()                                                .groupby('user_id')                                                .apply(lambda x: x.iloc[: k])                                                .rename(columns={'user_id': 'user_id_y'})                                                .groupby('user_id_y')['product_id_y'].agg(list)
    
        return out
    
    def _precision(self, arr1:np.ndarray, arr2:np.ndarray, n:int=None) -> float:
    
        """ Returns Precision@n score for two arrays """
        
        n = n or self._default_k
    
        return len(set(arr1) & set(arr2)) / n
    
    def evaluate(self, n:int=None) -> None:
        
        """ Calculates MAP@score for model. Works only with 'train' mode """
        
        if self.mode != self._modes[0]:
            print(f"Model in '{self._modes[1]}' mode cannot be evaluated!")
            return
        
        n = n or self._default_k
        
        assert isinstance(n, int), 'Wrong type of input!'
        
        pred = self.predict(users=None, k=n)
        df_test = pd.read_csv(self._model_eval_data)
        
        res = []
        for user, arr1 in zip(pred.index, pred.values):
            arr2 = df_test[df_test['user_id'].eq(user)]['product_id'].values
            res.append(precision(arr1, arr2))
    
        res = np.mean(res)
        print(f'MAP@{n} score: {res:.3f}')
        
    def make_submission(self, n:int=None) -> None:
        
        """ Creates submission file. Works only with 'inference' mode """
        
        if self.mode != self._modes[1]:
            print(f"Model in '{self._modes[0]}' mode cannot make submission!")
            return
        
        n = n or self._default_k
        
        assert isinstance(n, int), 'Wrong type of input!' 
        
        pred = self.predict(users=None, k=n)
        
        sub = pred.to_frame().reset_index().rename(columns={'user_id_y': 'user_id', 'product_id_y': 'product_id'})

        sub['product_id'] = sub['product_id'].apply(lambda x: ' '.join([str(elem) for elem in x]))

        sub.to_csv(self._model_sub_path, index=False)
        
        print('Submission succeed!')

