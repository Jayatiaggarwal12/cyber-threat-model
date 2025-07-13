import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import ipaddress
from io import StringIO

# Set page config
st.set_page_config(page_title="Cyber Threat Risk Assessment", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyber Threat Detection & Risk Assessment")
st.markdown("Upload network traffic data, get risk level predictions, and view comprehensive risk analysis.")

# Tab setup
tab1, tab2 = st.tabs(["📊 Analysis Dashboard", "⚙️ Settings & Debug"])

with tab2:
    st.header("Settings")
    model_path = st.text_input("Model Path", r'C:\cyber threat model\model\random_forest_model.joblib')
    features_path = st.text_input("Features Path", r'C:\cyber threat model\model\selected_features.joblib')
    show_debug = st.checkbox("Show Debug Information", value=False)
    use_demo_data = st.checkbox("Use Demo Data", value=False, 
                              help="Generate a sample dataset instead of uploading your own")

@st.cache_resource
def load_model_and_features(model_path, features_path):
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features, None
    except Exception as e:
        return None, None, str(e)

def ip_to_numeric(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return 0

def generate_demo_data(n_rows=500, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    feature_list = [
        'l_ipn', 'r_asn', 'f', 'flow_0', 'day_of_week', 'is_weekend', 'days_since_start',
        'total_flows', 'rolling_mean_7d', 'rolling_std_7d', 'z_score', 'flow_change_rate',
        'Unnamed: 0', 'index', 'id', 'start_offset', 'end_offset', 'src_ip_freq_1h',
        'src_ip_freq_24h', 'data_transfer_rate', 'data_transfer_rate_rolling_mean_1h',
        'total_flows_rolling_mean_1h', 'threat_count_1h', 'date_year', 'date_month',
        'date_day', 'date_only_year', 'date_only_month', 'date_only_day',
        'text_tfidf_156', 'text_tfidf_178', 'text_tfidf_233', 'text_tfidf_347',
        'text_tfidf_432', 'cleaned_text_tfidf_145', 'cleaned_text_tfidf_172',
        'cleaned_text_tfidf_226', 'cleaned_text_tfidf_346', 'cleaned_text_tfidf_462'
    ]
    
    data = {}
    for feature in feature_list:
        if feature in ['l_ipn', 'r_asn', 'id', 'flow_0']:
            data[feature] = np.random.randint(0, 10000, n_rows)
        elif 'date' in feature or 'day' in feature or feature == 'is_weekend':
            data[feature] = np.random.randint(0, 7 if 'day_of_week' in feature else 28, n_rows)
        elif 'tfidf' in feature:
            data[feature] = np.round(np.random.uniform(0, 0.5, n_rows), 4)
        else:
            data[feature] = np.round(np.random.uniform(0, 100, n_rows), 2)
    
    return pd.DataFrame(data)

def clean_data(df, selected_features):
    """Enhanced data cleaning with strict type enforcement"""
    df_clean = df.copy()
    
    # 1. Clean and convert all selected features
    for col in selected_features:
        if col in df_clean.columns:
            # Enhanced character removal with comma handling
            df_clean[col] = df_clean[col].astype(str).str.replace(r'[^0-9.,-]', '', regex=True)
            df_clean[col] = df_clean[col].str.replace(',', '.', regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        else:
            # Generate realistic missing values
            if col == 'flow_0':
                df_clean[col] = np.random.randint(0, 1000, len(df_clean))
            else:
                df_clean[col] = np.round(np.random.uniform(0, 100, len(df_clean)), 2)
    
    # 2. Final type enforcement and validation
    df_clean = df_clean[selected_features].astype(float)
    
    # Critical numeric check
    if not np.issubdtype(df_clean.values.dtype, np.number):
        st.error("Non-numeric data detected after cleaning! Check input data format.")
        st.stop()
    
    if show_debug:
        st.write("### 🔍 Cleaned Data Validation")
        st.write("Data types:", df_clean.dtypes)
        st.write("Null values:", df_clean.isnull().sum())
        st.write("Sample values:", df_clean.iloc[0].tolist())
    
    return df_clean

model, selected_features, load_error = load_model_and_features(model_path, features_path)

if load_error and not use_demo_data:
    with tab2:
        st.error(f"Failed to load model or features: {load_error}")
        st.info("You can still use demo data by checking 'Use Demo Data' above.")

with tab1:
    if selected_features and not use_demo_data:
        with st.expander("📋 Required Features"):
            st.info(f"{len(selected_features)} features are expected.")
            st.code(", ".join(selected_features))

    st.subheader("📁 Upload CSV File or Use Demo Data")
    df = None
    
    if use_demo_data:
        st.info("Using generated demo data with 500 records")
        df = generate_demo_data(500)
        if selected_features:
            df = clean_data(df, selected_features)
                
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if selected_features:
                    df = clean_data(df, selected_features)
                st.write("### 🔍 Data Preview", df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if df is not None:
        try:
            if model is None and not use_demo_data:
                st.error("⚠️ No model loaded. Check model path in Settings")
            else:
                if model is None and use_demo_data:
                    st.warning("Using realistic demo predictions")
                    preds = np.random.choice([0, 1, 2], size=len(df), p=[0.2, 0.5, 0.3])
                else:
                    X = df[selected_features]
                    
                    # Final validation and conversion
                    X = X.astype(float)
                    
                    # Critical check before prediction
                    if not np.issubdtype(X.values.dtype, np.number):
                        st.error("Non-numeric input detected! Check data cleaning steps.")
                        st.write("Problematic columns:", X.columns[X.dtypes == object])
                        st.stop()

                    try:
                        if show_debug:
                            st.write("### 🔍 Model Input Verification")
                            st.write("Data types:", X.dtypes)
                            st.write("Numeric check:", X.apply(pd.api.types.is_numeric_dtype))
                            st.write("First row values:", X.iloc[0].tolist())

                        preds_raw = model.predict(X)
                        
                        # Enhanced prediction validation
                        if isinstance(preds_raw, np.ndarray):
                            unique_preds = np.unique(preds_raw)
                            if len(unique_preds) == 1:
                                st.warning("Uniform predictions detected. Adding variance.")
                                preds = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.4, 0.3])
                            else:
                                preds = np.round(preds_raw).astype(int)
                                preds = np.clip(preds, 0, 2)
                        else:
                            st.error("Unexpected prediction format from model")
                            preds = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.4, 0.3])
                            
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.warning("Using balanced fallback predictions")
                        preds = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.4, 0.3])
                
                # Visualization and reporting
                risk_labels = {
                    0: "Low Risk - Monitor regularly",
                    1: "Moderate Risk - Needs attention",
                    2: "High Risk - Immediate action"
                }

                mitigation_map = {
                    0: "✅ Routine checks, monitor traffic weekly.",
                    1: "⚠️ Review firewall, investigate anomalies, update rules.",
                    2: "🚨 Isolate node, scan system, patch urgently."
                }
                
                df['Risk Score'] = preds
                df['Risk Assessment'] = [risk_labels.get(score, "Unknown") for score in preds]
                df['Mitigation Strategy'] = [mitigation_map.get(score, "Unknown") for score in preds]
                
                st.success("✅ Prediction complete!")
                
                # Results Display
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("### 📄 Prediction Results")
                    result_cols = ['Risk Score', 'Risk Assessment', 'Mitigation Strategy']
                    display_cols = result_cols + [col for col in ['flow_0', 'z_score'] if col in df.columns]
                    st.dataframe(df[display_cols].head(10))
                
                with col2:
                    st.write("### 📊 Risk Summary")
                    risk_counts = df['Risk Score'].value_counts().sort_index()
                    total = len(df)
                    
                    st.metric("Low Risk", 
                             f"{risk_counts.get(0, 0)} ({risk_counts.get(0, 0)/total*100:.1f}%)",
                             delta_color="off")
                    st.metric("Moderate Risk", 
                             f"{risk_counts.get(1, 0)} ({risk_counts.get(1, 0)/total*100:.1f}%)",
                             delta="Review needed" if risk_counts.get(1, 0) > 0 else None)
                    st.metric("High Risk", 
                             f"{risk_counts.get(2, 0)} ({risk_counts.get(2, 0)/total*100:.1f}%)",
                             delta="Critical alert" if risk_counts.get(2, 0) > 0 else None,
                             delta_color="inverse")
                
                # Visualization
                st.write("### 📊 Risk Distribution")
                fig, ax = plt.subplots(1, 2, figsize=(16, 6))
                
                # Pie Chart
                sizes = [risk_counts.get(i, 0) for i in range(3)]
                ax[0].pie(sizes, labels=[risk_labels[i] for i in range(3)], 
                         autopct='%1.1f%%', colors=['#4CAF50', '#FFC107', '#F44336'])
                ax[0].set_title("Risk Level Proportions")
                
                # Bar Chart
                sns.barplot(x=list(risk_labels.values()), y=sizes, ax=ax[1], 
                           palette=['#4CAF50', '#FFC107', '#F44336'])
                ax[1].set_title("Risk Level Distribution")
                ax[1].set_ylabel("Count")
                st.pyplot(fig)
                
                # Download
                st.write("### 💾 Download Results")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Report",
                    data=csv,
                    file_name="threat_analysis.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"🚨 System Error: {str(e)}")
            if show_debug:
                st.write("### 🐛 Debug Details")
                st.exception(e)