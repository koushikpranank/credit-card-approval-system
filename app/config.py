import os
from dotenv import load_dotenv

load_dotenv()

# Get the absolute path of the directory containing this config.py file
# Assuming config.py is inside the 'app' directory, we go one level up to get the project root.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Inside app/config.py
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Use a direct path straight to your project root folder
    SQLALCHEMY_DATABASE_URI = 'sqlite:////Users/koushikeslavath/DevTools/Credit/credit-card-approval-system/instance/approval.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'models/model_pipeline.pkl'
    DATA_PATH = os.environ.get('DATA_PATH') or 'data/credit_card_dataset.csv'
    PREDICTION_THRESHOLD = float(os.environ.get('THRESHOLD', 0.5))