# app1.py

import os
import io
import csv
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import plotly.express as px
from fpdf import FPDF

# =======================
# CONFIG / CONSTANTS
# =======================
DATA_FILE: str = "Housing.csv"
MODEL_FILE: str = "house_price_model.pkl"
USERS_FILE: str = "users.csv"
BACKGROUND_IMAGE_URL: str = "https://images.unsplash.com/photo-1560185127-6ed189bf02f4?auto=format&fit=crop&w=1920&q=60"

NUMERIC_COLS: list[str] = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_COLS: list[str] = ["mainroad", "guestroom", "basement", "hotwaterheating",
                               "airconditioning", "prefarea", "furnishingstatus"]

CURRENCY_RATES: dict[str, float] = {
    "PKR": 1.0,
    "USD": 0.0036,
    "EUR": 0.0033,
    "GBP": 0.0028,
    "AED": 0.013,
}

# =======================
# STYLE
# =======================
def set_background_from_url(image_url: str) -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stMainBlockContainer {{
            background: rgba(255,255,255,0.85);
            padding: 1.25rem 1rem;
            border-radius: 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =======================
# AUTH FUNCTIONS
# =======================
def ensure_users_file() -> None:
    """Create users CSV file if it doesn't exist with proper headers"""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password"])

def load_users() -> dict[str, str]:
    """Load users from CSV file into a dictionary"""
    ensure_users_file()
    users = {}
    try:
        with open(USERS_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "username" in row and "password" in row:
                    users[row["username"]] = row["password"]
    except (FileNotFoundError, csv.Error) as e:
        st.error(f"Error loading users: {e}")
    return users

def save_user(username: str, password: str) -> bool:
    """Save a new user to the CSV file if username doesn't exist"""
    users = load_users()
    if username in users:
        return False
    
    try:
        with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([username, password])
        return True
    except (IOError, csv.Error) as e:
        st.error(f"Error saving user: {e}")
        return False

def login_ui() -> bool:
    st.title("🏡 House Price Prediction — Login / Signup")
    tab_login, tab_signup = st.tabs(["🔐 Login", "🆕 Signup"])
    logged_in = False

    with tab_login:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="btn_login"):
            users = load_users()
            if login_user and users.get(login_user) == login_pass:
                st.session_state.logged_in = True
                st.session_state.current_user = login_user
                st.success(f"Welcome back, {login_user} 👋")
                logged_in = True
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        signup_user = st.text_input("Choose a username", key="signup_user")
        signup_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Create account", key="btn_signup"):
            if not signup_user or not signup_pass:
                st.warning("Please provide both username and password.")
            else:
                ok = save_user(signup_user, signup_pass)
                if ok:
                    st.success("Account created. You can log in now.")
                else:
                    st.error("Username already exists. Try a different one.")

    # Feature cards
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🤖 AI-POWERED**")
        st.markdown("ML algorithms for precise predictions")
    with col2:
        st.markdown("**📊 REAL-TIME ANALYSIS**")
        st.markdown("Instant property valuation with insights")
    with col3:
        st.markdown("**🚀 FUTURISTIC UI**")
        st.markdown("Advanced interface with immersive experience")

    return logged_in

# =======================
# DATA / MODEL
# =======================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess housing data with error handling."""
    if not os.path.exists(path):
        st.error(f"❌ Data file not found: {path}")
        st.stop()

    try:
        df = pd.read_csv(path)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        if "price" not in df.columns:
            st.error("❌ Required 'price' column not found in dataset.")
            st.stop()

        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load trained model from file path or BytesIO with error handling."""
    try:
        if isinstance(path, (io.BytesIO, bytes)):  # Handle BytesIO or bytes directly
            return joblib.load(path)
        elif isinstance(path, str) and os.path.exists(path):  # Handle local file path
            return joblib.load(path)
        else:
            st.error(f"❌ Model file not found or invalid: {path}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


# =======================
# UTILS
# =======================
def convert_currency_from_pkr(amount_pkr: float, currency: str) -> float:
    return float(amount_pkr) * CURRENCY_RATES.get(currency, 1.0)

def prepare_input_row(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Create input form for property details with smarter defaults"""
    st.subheader("📋 Enter House Details")
    col1, col2 = st.columns(2)
    ui_vals = {}

    # Calculate medians once instead of for each column
    numeric_defaults = {col: int(df[col].median()) for col in NUMERIC_COLS if col in df.columns}
    
    with col1:
        for col in NUMERIC_COLS:
            default_val = numeric_defaults.get(col, 0)
            ui_vals[col] = int(
                st.number_input(
                    col.replace("_", " ").title(),
                    value=default_val,
                    min_value=0,
                    step=1,
                    key=f"{prefix}_num_{col}"
                )
            )

    with col2:
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                # Use value_counts to get most common values first
                value_counts = df[col].value_counts()
                opts = value_counts.index.tolist()
            else:
                opts = ["no", "yes"] if col != "furnishingstatus" else ["unfurnished", "semi-furnished", "furnished"]
            
            ui_vals[col] = st.selectbox(
                col.replace("_", " ").title(),
                options=opts,
                key=f"{prefix}_cat_{col}"
            )

    return pd.DataFrame([ui_vals])

def encode_like_model(df_input: pd.DataFrame, model) -> pd.DataFrame:
    encoded = pd.get_dummies(df_input, drop_first=False)
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
    else:
        raise RuntimeError("Model does not expose feature_names_in_. Please retrain with scikit >=1.0.")
    encoded = encoded.reindex(columns=model_cols, fill_value=0)
    return encoded

def predict_with_confidence(model, X: pd.DataFrame, n_bootstraps: int = 100) -> tuple:
    """Generate prediction with confidence interval using bootstrapping"""
    if hasattr(model, 'predict'):
        predictions = []
        for _ in range(n_bootstraps):
            # Create bootstrap sample
            sample_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[sample_indices]
            
            # Predict
            pred = model.predict(X_sample)
            predictions.append(pred[0])
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        confidence_interval = (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred)
        
        return mean_pred, confidence_interval, std_pred
    
    return model.predict(X)[0], (0, 0), 0

def calculate_market_stats(df: pd.DataFrame, predicted_price: float, currency: str) -> dict:
    """Calculate how the predicted price compares to market statistics"""
    current_rate = CURRENCY_RATES[currency]
    market_prices = df['price'] * current_rate
    
    stats = {
        'percentile': round((predicted_price > market_prices).mean() * 100, 1),
        'market_avg': round(market_prices.mean(), 2),
        'market_median': round(market_prices.median(), 2),
        'market_min': round(market_prices.min(), 2),
        'market_max': round(market_prices.max(), 2),
        'price_difference_avg': round(predicted_price - market_prices.mean(), 2)
    }
    
    return stats

def find_similar_properties(df: pd.DataFrame, input_property: pd.DataFrame, 
                           n_recommendations: int = 3) -> pd.DataFrame:
    """Find similar properties in the dataset based on features"""
    # Select only relevant columns for comparison
    comparison_cols = NUMERIC_COLS + [col for col in CATEGORICAL_COLS if col in df.columns]
    
    # Create a copy and encode categorical variables
    df_encoded = df[comparison_cols].copy()
    input_encoded = input_property[comparison_cols].copy()
    
    # One-hot encode categorical variables
    for col in CATEGORICAL_COLS:
        if col in df_encoded.columns:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=False)
            input_encoded = pd.get_dummies(input_encoded, columns=[col], prefix=col, drop_first=False)
    
    # Align columns
    for col in df_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[df_encoded.columns]
    
    # Calculate similarity (Euclidean distance for numeric, exact match for categorical)
    distances = []
    for idx, row in df_encoded.iterrows():
        distance = np.sqrt(np.sum((row.values - input_encoded.values[0])**2))
        distances.append(distance)
    
    # Add distances to dataframe and return top matches
    df_with_dist = df.copy()
    df_with_dist['similarity_score'] = distances
    df_with_dist['similarity_rank'] = df_with_dist['similarity_score'].rank()
    
    return df_with_dist.nsmallest(n_recommendations, 'similarity_score')

def save_prediction_history(username: str, input_data: dict, prediction: float, currency: str) -> None:
    """Save prediction history for users"""
    history_file = f"history_{username}.csv"
    history_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': prediction,
        'currency': currency,
        **input_data
    }
    
    df_history = pd.DataFrame([history_data])
    
    if os.path.exists(history_file):
        df_history.to_csv(history_file, mode='a', header=False, index=False)
    else:
        df_history.to_csv(history_file, index=False)

def load_prediction_history(username: str) -> pd.DataFrame:
    """Load user's prediction history"""
    history_file = f"history_{username}.csv"
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    return pd.DataFrame()

# =======================
# PLOT FUNCTIONS
# =======================
def plot_comparison_stacked(prop1: pd.DataFrame, prop2: pd.DataFrame):
    combined = pd.DataFrame({
        'Feature': NUMERIC_COLS,
        'Property 1': [prop1[col].values[0] for col in NUMERIC_COLS],
        'Property 2': [prop2[col].values[0] for col in NUMERIC_COLS]
    })
    combined_melted = combined.melt(
        id_vars='Feature',
        value_vars=['Property 1', 'Property 2'],
        var_name='Property',
        value_name='Value'
    )
    fig = px.bar(
        combined_melted,
        x='Property',
        y='Value',
        color='Feature',
        text='Value',
        title="Stacked Comparison of Property Features",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        xaxis_title="Property",
        yaxis_title="Value",
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    fig.update_traces(textposition='inside')
    return fig

def plot_price_distribution(df: pd.DataFrame, predicted_price: float, currency: str) -> None:
    """Show where the predicted price falls in the overall market distribution"""
    current_rate = CURRENCY_RATES[currency]
    df_price_converted = df['price'] * current_rate
    
    fig = px.histogram(
        df_price_converted, 
        nbins=50,
        title=f"Market Price Distribution ({currency})",
        labels={'value': f'Price ({currency})', 'count': 'Number of Properties'}
    )
    
    # Add vertical line for predicted price
    fig.add_vline(
        x=predicted_price, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Predicted: {predicted_price:,.0f} {currency}",
        annotation_position="top"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(model, feature_names: list) -> None:
    """Show which features most influence the prediction"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance in Price Prediction',
            color='importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title='',
            xaxis_title='Importance'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type")

# =======================
# PDF FUNCTIONS
# =======================
def generate_pdf_text_only(username: str, currency: str, predicted_price: float, user_input_df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "House Price Prediction Report (Text Only)", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"User: {username}", ln=True)
    pdf.cell(0, 8, f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Predicted Price: {predicted_price:,.0f} {currency}", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", "", 11)
    for idx, row in user_input_df.iterrows():
        pdf.cell(0, 8, f"Property {idx+1} Details:", ln=True)
        for k, v in row.items():
            pdf.cell(0, 7, f" - {k.replace('_',' ').title()}: {v}", ln=True)
        pdf.ln(2)
    return pdf.output(dest="S").encode("latin1")


def generate_pdf_chart(username: str, currency: str, predicted_price: float, user_input_df: pd.DataFrame, chart_fig=None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "House Price Prediction Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"User: {username}", ln=True)
    pdf.cell(0, 8, f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Predicted Price: {predicted_price:,.0f} {currency}", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", "", 11)
    for idx, row in user_input_df.iterrows():
        pdf.cell(0, 8, f"Property {idx+1} Details:", ln=True)
        for k, v in row.items():
            pdf.cell(0, 7, f" - {k.replace('_',' ').title()}: {v}", ln=True)
        pdf.ln(2)
    if chart_fig is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            try:
                chart_fig.write_image(tmp.name, format="png", scale=2)
                pdf.image(tmp.name, x=10, w=180)
                os.unlink(tmp.name)
            except Exception as e:
                pdf.cell(0, 7, f"Chart could not be generated: {e}", ln=True)
    return pdf.output(dest="S").encode("latin1")


# =======================
# CHATGPT-LIKE SIDEBAR
# =======================
def chatgpt_sidebar():
    with st.sidebar:
        st.markdown("## 🏠 House Price App")
        
        if st.button("➕ New Prediction"):
            st.rerun()
        
        # Prediction history
        if st.session_state.get('current_user'):
            with st.expander("📋 Prediction History", expanded=False):
                history = load_prediction_history(st.session_state.current_user)
                if not history.empty:
                    st.dataframe(history.tail(5), use_container_width=True)
                    if st.button("Clear History"):
                        os.remove(f"history_{st.session_state.current_user}.csv")
                        st.rerun()
                else:
                    st.info("No prediction history yet")
        
        with st.expander("ℹ️ About / System Info", expanded=False):
            st.markdown("**Developer:** Abu Sufyan")
            st.markdown("**Email:** mrabusufyan2003@gmail.com")
            st.markdown("**Institute:** University of Narowal")
            st.markdown(f"**Python Version:** {os.sys.version.split()[0]}")
            st.markdown(f"**OS:** {os.name}")
            st.markdown(f"**Current User:** {st.session_state.get('current_user', 'Guest')}")
        
        st.markdown("---")
        st.markdown("👤")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()

# =======================
# APP MAIN
# =======================
def app_main(username: str, df: pd.DataFrame, model):
    st.header(f"🏠 House Price Prediction — Welcome {username}")
    currency = st.selectbox("Select Currency", list(CURRENCY_RATES.keys()), index=0)

    with st.expander("📋 Tips before entering details", expanded=True):
        st.markdown("""
        - **Numeric**: Area, Bedrooms, Bathrooms, Stories, Parking are integers
        - **Categorical**: Yes/No fields and Furnishing Status remain as strings
        - **Predictions** are internally in PKR, converted to selected currency
        - Use realistic values; model learns patterns, not guarantees
        - **New**: See how your property compares to market trends
        """)

    pred_type = st.radio("Choose Prediction Type", ["Single Property", "Compare Two Properties"], horizontal=True)

    if pred_type == "Single Property":
        user_row = prepare_input_row(df, prefix="single")
        
        # Add confidence level option
        confidence_analysis = st.checkbox("📊 Show advanced confidence analysis", value=False)
        
        if st.button("Predict Price", key="single_predict"):
            try:
                encoded = encode_like_model(user_row, model)
                
                if confidence_analysis:
                    pred_price, confidence_interval, std_dev = predict_with_confidence(model, encoded)
                    st.success(f"💰 Predicted Price: {pred_price:,.0f}")
                    st.info(f"📈 95% Confidence Interval: {confidence_interval[0]:,.0f} - {confidence_interval[1]:,.0f}")
                    st.info(f"📊 Standard Deviation: ±{std_dev:,.0f}")
                else:
                    pred_price = float(model.predict(encoded)[0])
                    st.success(f"💰 Predicted Price: {pred_price:,.0f}")
                
                pred_price_curr = convert_currency_from_pkr(pred_price, currency)
                
                # Save prediction history
                save_prediction_history(username, user_row.iloc[0].to_dict(), pred_price_curr, currency)
                
                # Market comparison stats
                market_stats = calculate_market_stats(df, pred_price_curr, currency)
                st.subheader("📊 Market Comparison")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Percentile Rank", f"{market_stats['percentile']}%")
                    st.metric("Market Average", f"{market_stats['market_avg']:,.0f} {currency}")
                with col2:
                    st.metric("Market Median", f"{market_stats['market_median']:,.0f} {currency}")
                    st.metric("Price vs Average", f"{market_stats['price_difference_avg']:,.0f} {currency}")
                with col3:
                    st.metric("Market Range", f"{market_stats['market_min']:,.0f} - {market_stats['market_max']:,.0f} {currency}")

                # Price distribution plot
                plot_price_distribution(df, pred_price_curr, currency)
                
                # Feature importance plot
                if hasattr(model, 'feature_importances_'):
                    plot_feature_importance(model, encoded.columns.tolist())

                # Similar properties recommendation
                st.subheader("🏘️ Similar Properties in Market")
                similar_properties = find_similar_properties(df, user_row)
                st.dataframe(
                    similar_properties[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'furnishingstatus']].assign(
                        price=lambda x: x['price'] * CURRENCY_RATES[currency]
                    ).round(0),
                    use_container_width=True
                )

                # Single property stacked bar
                single_fig = px.bar(
                    x=NUMERIC_COLS,
                    y=[user_row[col].values[0] for col in NUMERIC_COLS],
                    text=[user_row[col].values[0] for col in NUMERIC_COLS],
                    labels={"x": "Feature", "y": "Value"},
                    title="Property Feature Values (Single Property)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                single_fig.update_traces(textposition='inside')
                single_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(single_fig, use_container_width=True)

                pdf_text_bytes = generate_pdf_text_only(username, currency, pred_price_curr, user_row)
                pdf_chart_bytes = generate_pdf_chart(username, currency, pred_price_curr, user_row, chart_fig=single_fig)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📄 Download Prediction Report (Text PDF)", data=pdf_text_bytes,
                                       file_name=f"House_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mime="application/pdf")
                with col2:
                    st.download_button("📄 Download Prediction Report (Chart PDF)", data=pdf_chart_bytes,
                                       file_name=f"House_Prediction_Chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mime="application/pdf")

            except Exception as e:
                st.error(f"Error: {e}")
                st.error("Please check your input values and try again.")

    elif pred_type == "Compare Two Properties":
        st.subheader("Property 1 Details")
        prop1 = prepare_input_row(df, prefix="prop1")
        st.subheader("Property 2 Details")
        prop2 = prepare_input_row(df, prefix="prop2")

        if st.button("Predict & Compare", key="compare_predict"):
            try:
                enc1 = encode_like_model(prop1, model)
                enc2 = encode_like_model(prop2, model)
                price1 = float(model.predict(enc1)[0])
                price2 = float(model.predict(enc2)[0])
                price1_curr = convert_currency_from_pkr(price1, currency)
                price2_curr = convert_currency_from_pkr(price2, currency)

                st.success(f"💰 Property 1: {price1_curr:,.0f} {currency}")
                st.success(f"💰 Property 2: {price2_curr:,.0f} {currency}")

                combined_df = pd.concat([prop1, prop2], ignore_index=True)
                fig_bar = plot_comparison_stacked(prop1, prop2)
                st.plotly_chart(fig_bar, use_container_width=True)

                pdf_text_bytes = generate_pdf_text_only(username, currency, max(price1_curr, price2_curr), combined_df)
                pdf_chart_bytes = generate_pdf_chart(username, currency, max(price1_curr, price2_curr), combined_df, chart_fig=fig_bar)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📄 Download Comparison Report (Text PDF)", data=pdf_text_bytes,
                                       file_name=f"House_Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mime="application/pdf")
                with col2:
                    st.download_button("📄 Download Comparison Report (Chart PDF)", data=pdf_chart_bytes,
                                       file_name=f"House_Comparison_Chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mime="application/pdf")

            except Exception as e:
                st.error(f"Error: {e}")

# =======================
# MAIN
# =======================
def main() -> None:
    """Main application entry point"""
    st.set_page_config(page_title="House Price Prediction", layout="wide", page_icon="🏠")
    set_background_from_url(BACKGROUND_IMAGE_URL)
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    # Load data and model
    df = load_data(DATA_FILE)
    model = load_model(MODEL_FILE)

    
    # Authentication flow
    if not st.session_state.logged_in:
        if login_ui():
            st.rerun()
    else:
        chatgpt_sidebar()
        app_main(st.session_state.current_user, df, model)

if __name__ == "__main__":
    main()