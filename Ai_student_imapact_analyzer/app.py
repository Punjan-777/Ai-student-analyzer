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
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Student Impact Analyzer", layout="wide")
st.title("AI Student Impact Analyzer")

# ------------------------- FIND CSV AUTOMATICALLY -------------------------
def find_csv():
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.lower().startswith("ai_student_impact_dataset") and f.endswith(".csv"):
                return os.path.join(root, f)
    return None

csv_path = find_csv()
if csv_path is None:
    st.error("CSV file not found. Please upload manually.")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(csv_path)

st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------- PREPROCESS -------------------------
def preprocess(df):
    df2 = df.copy()
    encoders = {}
    for col in df2.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col].astype(str))
        encoders[col] = le
    targets = [c for c in ['Post_Semester_GPA', 'Skill_Retention_Score', 'Burnout_Risk_Level'] if c in df2.columns]
    features = [c for c in df2.columns if c not in targets]
    return df2, features, targets, encoders

df2, features, targets, encoders = preprocess(df)

# ------------------------- SIDEBAR -------------------------
target = st.sidebar.selectbox("Select Target", targets)
task = "regression" if target != 'Burnout_Risk_Level' else "classification"
model_type = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost", "Deep Learning"])

if st.sidebar.button("Train and Evaluate"):
    X = df2[features]
    y = df2[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----- Train Model -----
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42) if task=="regression" else RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42) if task=="regression" else XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:  # Deep Learning
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
        except:
            st.error("TensorFlow not installed. Please add `tensorflow-cpu` to requirements.txt")
            st.stop()
        num_classes = None if task=="regression" else len(np.unique(y))
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        if task == "regression":
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse')
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
        if task == "regression":
            y_pred = model.predict(X_test_scaled).flatten()
        else:
            y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

    # ----- Store in session -----
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['features'] = features
    st.session_state['df_raw'] = df
    st.session_state['target'] = target
    st.session_state['task'] = task
    st.session_state['model_type'] = model_type
    st.session_state['encoders'] = encoders

    # ----- Show results -----
    st.header("Evaluation Results")
    if task == "regression":
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{rmse:.4f}")
        c2.metric("R2", f"{r2:.4f}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        st.pyplot(fig)
    else:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=imp, x='Importance', y='Feature')
        st.pyplot(fig)

# ------------------------- LIVE PREDICTION -------------------------
st.header("Live Prediction")
if 'model' not in st.session_state:
    st.info("Train a model first")
else:
    with st.form("pred_form"):
        st.subheader("Adjust student attributes")
        input_data = {}
        cols = st.columns(3)
        for i, col in enumerate(st.session_state['features']):
            with cols[i % 3]:
                raw = st.session_state['df_raw'][col]
                if raw.dtype == 'object' or raw.nunique() < 10:
                    options = sorted(raw.dropna().unique())
                    input_data[col] = st.selectbox(col, options)
                else:
                    minv = float(raw.min()); maxv = float(raw.max())
                    default = float(raw.median())
                    step = round((maxv - minv) / 100, 2) or 0.1
                    input_data[col] = st.slider(col, minv, maxv, default, step)

        if st.form_submit_button("Predict"):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            # Encode categoricals using stored encoders
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    le = st.session_state['encoders'][col]
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except:
                        input_df[col] = 0
            # Ensure column order
            input_df = input_df[st.session_state['features']]
            # Predict
            if st.session_state['model_type'] == "Deep Learning":
                input_scaled = st.session_state['scaler'].transform(input_df)
                pred = st.session_state['model'].predict(input_scaled)
                if st.session_state['task'] == "regression":
                    pred = pred[0][0]
                else:
                    pred = np.argmax(pred, axis=1)[0]
            else:
                pred = st.session_state['model'].predict(input_df)[0]
            # Display
            if st.session_state['task'] == "regression":
                st.metric(f"Predicted {st.session_state['target']}", f"{pred:.3f}")
            else:
                if st.session_state['target'] == 'Burnout_Risk_Level':
                    labels = sorted(st.session_state['df_raw']['Burnout_Risk_Level'].dropna().unique())
                    pred_label = labels[int(pred)] if 0 <= int(pred) < len(labels) else f"Class {int(pred)}"
                    st.metric("Predicted Burnout Risk", pred_label)
                else:
                    st.metric(f"Predicted {st.session_state['target']}", f"Class {int(pred)}")
