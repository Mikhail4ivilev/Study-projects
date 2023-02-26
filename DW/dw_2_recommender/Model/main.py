
import os

from recommend_database import RecommendDataBase
from recommend_model import RecommendModel

import warnings
warnings.filterwarnings('ignore')

cur_dir = os.getcwd()

db = RecommendDataBase(directory=cur_dir)
db.prepare_model_data()

model = RecommendModel(database=db, mode='inference')

model.train(full=True, k=10)

model.make_submission(n=10)

