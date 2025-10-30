import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="HotelSmart - Cancellation Prediction",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🏨 HotelSmart - Cancellation Prediction System")
st.markdown("---")

# Loading saved models
@st.cache_resource
def load_models():
    try:
        with open('model/final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('parameter/hotelsmart_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('parameter/market_segment_type_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("Models not found! Make sure the .pkl files are in the correct directory.")
        return None, None, None

model, scaler, label_encoder = load_models()

if model is not None:
    # Sidebar for data input
    st.sidebar.header("📊 Reservation Data")

    # User inputs
    lead_time = st.sidebar.number_input(
        "Lead Time (days in advance)",
        min_value=0,
        max_value=500,
        value=30,
        help="Number of days between booking and arrival"
    )

    arrival_month = st.sidebar.selectbox(
        "Arrival Month",
        options=list(range(1, 13)),
        index=5,
        help="Month of arrival (1-12)"
    )

    arrival_date = st.sidebar.number_input(
        "Arrival Date (day of month)",
        min_value=1,
        max_value=31,
        value=15,
        help="Day of the month of arrival"
    )

    market_segment_type = st.sidebar.selectbox(
        "Market Segment Type",
        options=["Aviation", "Complementary", "Corporate", "Online", "Offline"],
        index=3,
        help="Booking market segment"
    )

    avg_price_per_room = st.sidebar.number_input(
        "Average Price per Room ($)",
        min_value=0.0,
        max_value=10000.0,
        value=150.0,
        step=10.0,
        help="Average price per room in dollars"
    )

    no_of_special_requests = st.sidebar.number_input(
        "Number of Special Requests",
        min_value=0,
        max_value=10,
        value=1,
        help="Number of special requests made by the guest"
    )

    # Prediction button
    if st.sidebar.button("🔮 Make Prediction", type="primary"):
        # Prepare data for prediction
        try:
            # Encode market_segment_type
            market_segment_encoded = label_encoder.transform([[market_segment_type]])[0][0]

            # Create DataFrame with input data
            input_data = pd.DataFrame({
                'lead_time': [lead_time],
                'arrival_month': [arrival_month],
                'arrival_date': [arrival_date],
                'market_segment_type': [market_segment_encoded],
                'avg_price_per_room': [avg_price_per_room],
                'no_of_special_requests': [no_of_special_requests]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Prediction Result")
                if prediction == 1:
                    st.error("⚠️ **HIGH CANCELLATION PROBABILITY**")
                    st.markdown(f"**Cancellation Probability:** {prediction_proba[1]:.2%}")
                else:
                    st.success("✅ **LOW CANCELLATION PROBABILITY**")
                    st.markdown(f"**Retention Probability:** {prediction_proba[0]:.2%}")

            with col2:
                st.subheader("📊 Probabilities")
                prob_df = pd.DataFrame({
                    'Status': ['No Cancellation', 'Cancellation'],
                    'Probability': [prediction_proba[0], prediction_proba[1]]
                })
                st.bar_chart(prob_df.set_index('Status'))
            
            # Additional information
            st.markdown("---")
            st.subheader("💡 Recommendations")

            if prediction == 1:
                st.warning("""
                **Recommended Actions to Reduce Cancellation Risk:**
                - Contact the customer to confirm the reservation
                - Offer flexibility in dates or conditions
                - Check if there are unmet special needs
                - Consider offers or upgrades to retain the customer
                """)
            else:
                st.info("""
                **Reservation with Low Cancellation Risk:**
                - Customer will likely maintain the reservation
                - Focus on providing an excellent experience
                - Prepare adequately for the guest's arrival
                """)
            
            # Analysis details
            with st.expander("🔍 Analysis Details"):
                st.write("**Input Data:**")
                st.json({
                    "Lead Time": f"{lead_time} days",
                    "Arrival Month": arrival_month,
                    "Arrival Date": arrival_date,
                    "Market Segment": market_segment_type,
                    "Average Price per Room": f"$ {avg_price_per_room:.2f}",
                    "Special Requests": no_of_special_requests
                })

                st.write("**Influencing Factors:**")
                st.markdown("""
                - **Lead Time**: Bookings made far in advance tend to have higher cancellation risk
                - **Price**: Higher prices may influence cancellation decisions
                - **Special Requests**: Customers with special requests tend to cancel less
                - **Market Segment**: Different segments have distinct behaviors
                """)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Model information
    st.markdown("---")
    st.subheader("ℹ️ About the Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Random Forest")

    with col2:
        st.metric("Features Used", "6")

    with col3:
        st.metric("Estimated Accuracy", "~85%")
    
    with st.expander("📋 Technical Information"):
        st.markdown("""
        **Model Characteristics:**
        - **Algorithm**: Random Forest Classifier
        - **Features**: lead_time, arrival_month, arrival_date, market_segment_type, avg_price_per_room, no_of_special_requests
        - **Preprocessing**: StandardScaler for numeric variables, LabelEncoder for categorical variables
        - **Feature Selection**: Based on the Boruta algorithm

        **How to Use:**
        1. Fill in the reservation data in the sidebar
        2. Click "Make Prediction"
        3. Analyze the result and recommendations
        4. Take preventive actions if necessary
        """)

else:
    st.error("⚠️ Could not load the models. Check if the files are in the correct directory.")
    st.info("""
    **Required files:**
    - model/final_model.pkl
    - parameter/hotelsmart_scaler.pkl
    - parameter/market_segment_type_encoder.pkl
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏨 HotelSmart - Cancellation Prediction System | Developed with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)