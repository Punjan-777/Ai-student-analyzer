# config.py
# Global settings and constants

# Paths
DATA_PATH = "ai_student_impact_dataset(1).csv"
MODEL_DIR = "models"

# Target columns recognized in the dataset
TARGET_COLUMNS = ['Post_Semester_GPA', 'Skill_Retention_Score', 'Burnout_Risk_Level']

# Task type mapping
TASK_TYPES = {
    'Post_Semester_GPA': 'regression',
    'Skill_Retention_Score': 'regression',
    'Burnout_Risk_Level': 'classification'
}

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Deep Learning settings
DL_EPOCHS = 50
DL_BATCH_SIZE = 32
