DATA_PATH = "ai_student_impact_dataset (1).csv"
TARGET_COLUMNS = ['Post_Semester_GPA', 'Skill_Retention_Score', 'Burnout_Risk_Level']
TASK_TYPES = {
    'Post_Semester_GPA': 'regression',
    'Skill_Retention_Score': 'regression',
    'Burnout_Risk_Level': 'classification'
}
RANDOM_STATE = 42
TEST_SIZE = 0.2
DL_EPOCHS = 50
DL_BATCH_SIZE = 32
