# model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import config

def train_model(X, y, model_type, task_type):
    """
    Train model: Random Forest, XGBoost, or Deep Learning.
    Returns: model, scaler, X_test, y_test, y_pred, history
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = None
    y_pred = None
    history = None

    if model_type == "Random Forest":
        if task_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == "XGBoost":
        if task_type == "regression":
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=config.RANDOM_STATE)
        else:
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=config.RANDOM_STATE)
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
            epochs=config.DL_EPOCHS,
            batch_size=config.DL_BATCH_SIZE,
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
    """Predict on a single row (DataFrame)."""
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