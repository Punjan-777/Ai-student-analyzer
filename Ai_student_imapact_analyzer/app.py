import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIG ----------
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

# ---------- FIND CSV AUTOMATICALLY ----------
def find_csv():
    possible_paths = [
        "ai_student_impact_dataset (1).csv",
        "ai_student_impact_dataset.csv",
        "data/ai_student_impact_dataset (1).csv",
        "data/ai_student_impact_dataset.csv",
        "Ai_student_impact_analyzer/data/ai_student_impact_dataset (1).csv",
        "Ai_student_impact_analyzer/data/ai_student_impact_dataset.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# ---------- FUNCTIONS ----------
def load_data():
    path = find_csv()
    if path:
        return pd.read_csv(path)
    return None

def preprocess_data(df):
    df_processed = df.copy()
    le = LabelEncoder()
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    existing_targets = [col for col in TARGET_COLUMNS if col in df_processed.columns]
    feature_cols = [col for col in df_processed.columns if col not in existing_targets]
    return df_processed, feature_cols, existing_targets

def train_model(X, y, model_type, task_type):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = None
    y_pred = None
    history = None

    if model_type == "Random Forest":
        if task_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == "XGBoost":
        if task_type == "regression":
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
        else:
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == "Deep Learning":
        num_classes = None
        if task_type == "classification":
            num_classes = len(np.unique(y))
        
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        
        if task_type == "regression":
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            if num_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(Dense(num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train_scaled, y_train,
            epochs=DL_EPOCHS,
            batch_size=DL_BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        if task_type == "regression":
            y_pred = model.predict(X_test_scaled).flatten()
        else:
            y_pred_proba = model.predict(X_test_scaled)
            if num_classes == 2:
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
    
    return model, scaler, X_test, y_test, y_pred, history

def predict_single(model, scaler, input_data, model_type, task_type):
    if model_type == "Deep Learning":
        input_scaled = scaler.transform(input_data)
        if task_type == "regression":
            return float(model.predict(input_scaled, verbose=0)[0][0])
        else:
            pred_proba = model.predict(input_scaled, verbose=0)
            if pred_proba.shape[1] == 1:
                return int((pred_proba > 0.5).astype(int)[0][0])
            else:
                return int(np.argmax(pred_proba, axis=1)[0])
    else:
        return float(model.predict(input_data)[0])

def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    return fig

def plot_feature_importance(feature_names, importances, top_n=10):
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances')
    return fig

def plot_confusion_matrix(cm, labels=None):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Student Impact Analyzer", layout="wide")
st.title("🎓 AI Student Impact Analyzer")
st.markdown("**Analyze how Generative AI usage affects GPA, Skill Retention, and Burnout Risk.**")

# ---------- DATA LOADER ----------
st.sidebar.header(" 1. Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload CSV (or auto-load)", type=["csv"])

@st.cache_data
def get_data():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    df = load_data()
    if df is None:
        st.sidebar.error("CSV not found. Please upload manually.")
        return None
    return df

df = get_data()
if df is None:
    st.stop()

st.sidebar.success(f" Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
with st.expander(" Preview Raw Data"):
    st.dataframe(df.head(10))

# ---------- PREPROCESS ----------
@st.cache_data
def process(df):
    return preprocess_data(df)

df_processed, feature_cols, target_cols = process(df)

# ---------- MODEL SETUP ----------
st.sidebar.header(" 2. Model Setup")
target_choice = st.sidebar.selectbox("Select Target Variable", target_cols)
task_type = TASK_TYPES.get(target_choice, "regression")
st.sidebar.write(f"**Task Type:** {task_type.capitalize()}")

model_choice = st.sidebar.selectbox(
    "Select Algorithm",
    ["Random Forest", "XGBoost", "Deep Learning"]
)

# ---------- TRAIN ----------
if st.sidebar.button(" Train & Evaluate"):
    with st.spinner(f"Training {model_choice} on {target_choice}..."):
        X = df_processed[feature_cols]
        y = df_processed[target_choice]
        model, scaler, X_test, y_test, y_pred, history = train_model(X, y, model_choice, task_type)
        
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['model_type'] = model_choice
        st.session_state['task_type'] = task_type
        st.session_state['feature_cols'] = feature_cols
        st.session_state['df_raw'] = df
        st.session_state['target'] = target_choice
        st.session_state['df_processed'] = df_processed
        
        st.header(" Evaluation Results")
        col1, col2 = st.columns(2)
        if task_type == "regression":
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")
            st.pyplot(plot_actual_vs_predicted(y_test, y_pred))
        else:
            acc = accuracy_score(y_test, y_pred)
            col1.metric("Accuracy", f"{acc:.4f}")
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            cm = confusion_matrix(y_test, y_pred)
            labels = None
            if target_choice in df.columns:
                labels = sorted(df[target_choice].dropna().unique())
            st.pyplot(plot_confusion_matrix(cm, labels))
        
        if model_choice in ["Random Forest", "XGBoost"] and hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importances")
            st.pyplot(plot_feature_importance(feature_cols, model.feature_importances_))

# ---------- LIVE PREDICTION ----------
st.header("🧪 Live Prediction")
if 'model' not in st.session_state:
    st.info(" Please train a model first.")
else:
    with st.form("prediction_form"):
        st.subheader("Enter Student Attributes")
        input_data = {}
        cols = st.columns(3)
        for idx, col_name in enumerate(st.session_state['feature_cols']):
            with cols[idx % 3]:
                if st.session_state['df_raw'][col_name].dtype == 'object' or st.session_state['df_raw'][col_name].nunique() < 10:
                    options = sorted(st.session_state['df_raw'][col_name].dropna().unique())
                    input_data[col_name] = st.selectbox(f"{col_name}", options, key=f"pred_{col_name}")
                else:
                    min_val = float(st.session_state['df_raw'][col_name].min())
                    max_val = float(st.session_state['df_raw'][col_name].max())
                    default = float(st.session_state['df_raw'][col_name].median())
                    step = round((max_val - min_val) / 100, 2) or 0.1
                    input_data[col_name] = st.slider(f"{col_name}", min_val, max_val, default, step, key=f"pred_{col_name}")
        
        if st.form_submit_button("Predict Outcome"):
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            # Encode categoricals
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(st.session_state['df_raw'][col].astype(str))
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except:
                        input_df[col] = 0
            # Ensure columns are in the same order as training
            input_df = input_df[st.session_state['feature_cols']]
            
            # Predict
            pred = predict_single(
                st.session_state['model'],
                st.session_state['scaler'],
                input_df,
                st.session_state['model_type'],
                st.session_state['task_type']
            )
            
            st.subheader(" Prediction Result")
            if st.session_state['task_type'] == "regression":
                st.metric(f"Predicted {st.session_state['target']}", f"{pred:.3f}")
            else:
                if st.session_state['target'] == 'Burnout_Risk_Level':
                    labels = sorted(st.session_state['df_raw']['Burnout_Risk_Level'].dropna().unique())
                    pred_label = labels[int(pred)] if 0 <= int(pred) < len(labels) else f"Class {int(pred)}"
                    st.metric("Predicted Burnout Risk", pred_label)
                else:
                    st.metric(f"Predicted {st.session_state['target']}", f"Class {int(pred)}")

st.caption("Built with ❤️ using Streamlit | Dataset: AI Impact on Students (Kaggle)")
