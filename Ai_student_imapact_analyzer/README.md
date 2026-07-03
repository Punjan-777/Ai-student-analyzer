# AI Student Impact Analyzer

A comprehensive AI-powered interactive dashboard that analyzes how Generative AI usage impacts student performance (GPA), skill retention, and burnout risk. Supports Random Forest, XGBoost, and Deep Learning models.

## Features
- Multi-target prediction (GPA, Skill Retention, Burnout Risk)
- Hybrid ML/DL backend
- Interactive EDA dashboard
- Live prediction with sliders/dropdowns
- Feature importance explainability

## Tech Stack
- **Frontend:** Streamlit
- **ML/DL:** Scikit-learn, XGBoost, TensorFlow/Keras
- **Visualization:** Matplotlib, Seaborn

## Installation & Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place the dataset in `data/` folder
4. Run: `streamlit run app.py`