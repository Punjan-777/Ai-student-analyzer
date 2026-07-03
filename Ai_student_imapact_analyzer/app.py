# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

import config
from data_loader import load_data, preprocess_data, get_original_labels
from model import train_model, predict_single
from utils import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_actual_vs_predicted
)

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Student Impact Analyzer", layout="wide")
st.title("AI Student Impact Analyzer")
st.markdown("**Analyze how Generative AI usage affects GPA, Skill Retention, and Burnout Risk.**")

# ------------------- SIDEBAR: DATA LOADER -------------------
st.sidebar.header("1. Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload CSV (or use default)", type=["csv"])

@st.cache_data
def get_data(file):
    if file is not None:
        return pd.read_csv(file)
    df = load_data()
    if df is None:
        st.sidebar.error("Default dataset not found. Please upload a CSV.")
        st.stop()
    return df

df = get_data(uploaded_file)
st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

with st.expander("Preview Raw Data"):
    st.dataframe(df.head(10))

# ------------------- PREPROCESS -------------------
@st.cache_data
def process(df):
    return preprocess_data(df)

df_processed, feature_cols, target_cols = process(df)

# ------------------- SIDEBAR: MODEL SETUP -------------------
st.sidebar.header("2. Model Setup")
target_choice = st.sidebar.selectbox("Select Target Variable", target_cols)
task_type = config.TASK_TYPES.get(target_choice, "regression")
st.sidebar.write(f"**Task Type:** {task_type.capitalize()}")

model_choice = st.sidebar.selectbox(
    "Select Algorithm",
    ["Random Forest", "XGBoost", "Deep Learning"]
)

# ------------------- TRAIN & EVALUATE -------------------
if st.sidebar.button("Train & Evaluate"):
    with st.spinner(f"Training {model_choice} on {target_choice}..."):
        X = df_processed[feature_cols]
        y = df_processed[target_choice]
        
        model, scaler, X_test, y_test, y_pred, history = train_model(
            X, y, model_choice, task_type
        )
        
        # Store in session state for live prediction
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['model_type'] = model_choice
        st.session_state['task_type'] = task_type
        st.session_state['feature_cols'] = feature_cols
        st.session_state['df_raw'] = df
        st.session_state['target'] = target_choice
        
        # ---------- EVALUATION RESULTS ----------
        st.header("Evaluation Results")
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
        
        # Feature Importance
        if model_choice in ["Random Forest", "XGBoost"] and hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importances")
            st.pyplot(plot_feature_importance(feature_cols, model.feature_importances_))

# ------------------- LIVE PREDICTION -------------------
st.header("Live Prediction")
if 'model' not in st.session_state:
    st.info("Please train a model first using the sidebar above.")
else:
    with st.form("prediction_form"):
        st.subheader("Enter Student Attributes")
        input_data = {}
        cols = st.columns(3)
        
        for idx, col_name in enumerate(st.session_state['feature_cols']):
            with cols[idx % 3]:
                unique_vals = st.session_state['df_raw'][col_name].nunique()
                if unique_vals < 10 or st.session_state['df_raw'][col_name].dtype == 'object':
                    options = sorted(st.session_state['df_raw'][col_name].dropna().unique())
                    input_data[col_name] = st.selectbox(f"{col_name}", options, key=f"pred_{col_name}")
                else:
                    min_val = float(st.session_state['df_raw'][col_name].min())
                    max_val = float(st.session_state['df_raw'][col_name].max())
                    default = float(st.session_state['df_raw'][col_name].median())
                    step = round((max_val - min_val) / 100, 2) or 0.1
                    input_data[col_name] = st.slider(
                        f"{col_name}", min_val, max_val, default, step, key=f"pred_{col_name}"
                    )
        
        if st.form_submit_button("Predict Outcome"):
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
            input_df = input_df[st.session_state['feature_cols']]
            
            pred = predict_single(
                st.session_state['model'], st.session_state['scaler'],
                input_df, st.session_state['model_type'], st.session_state['task_type']
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

st.caption("Built with using Streamlit | Dataset: AI Impact on Students (Kaggle)")