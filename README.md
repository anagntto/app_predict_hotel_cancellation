# 🏨 HotelSmart - Booking Cancellation Prediction System

## 📌 Overview

**HotelSmart** is a machine learning-based web application built to predict hotel booking cancellations. It leverages a pre-trained **Random Forest Classifier**, derived from historical booking data, to estimate the probability of a reservation being canceled.

This app was developed using the full machine learning pipeline from the [hotel_cancellation_predict](https://github.com/anagntto/hotel_cancellation_predict) project.

## 🚀 About the App

This Streamlit-based app allows hotel managers to input reservation details and receive an immediate cancellation risk prediction. Based on the model output, the app provides actionable recommendations to minimize cancellations and optimize revenue.

## 🧠 Model Characteristics

- **Algorithm**: Random Forest Classifier  
- **Main Features Used**: 6 key variables  
- **Estimated Accuracy**: ~85%  
- **Preprocessing**: StandardScaler for numeric variables, LabelEncoder for categorical encoding  

### 🔍 Features Used in the Model

1. **Lead Time** – Number of days between reservation and check-in  
2. **Arrival Month** – Month of arrival  
3. **Arrival Date** – Day of the month of arrival  
4. **Market Segment Type** – Type of booking segment (e.g., Online, Corporate)  
5. **Average Price per Room** – Price in local currency  
6. **Number of Special Requests** – Count of special service requests  

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit** for frontend interaction
- **scikit-learn** for model training and inference
- **pandas**, **numpy** for data manipulation
- **pickle** for model and transformer serialization

## 🌐 App Demo

👉 Try the live demo here:  
[https://apppredict-cgwndpfq8n454uzb6gcccw.streamlit.app/](https://apppredict-cgwndpfq8n454uzb6gcccw.streamlit.app/)

## 🔄 Related Project

This app is built from the model and pipeline developed in the main notebook from:  
📁 [hotel_cancellation_predict](https://github.com/anagntto/hotel_cancellation_predict)

That project includes:
- Business Case definition
- Full EDA
- Feature Engineering
- Feature Selection (Tree importance, Lasso, Boruta)
- Model comparison and tuning (Random Forest, XGBoost, LightGBM, etc.)

## 👨‍💻 Author

Developed by [@anagntto](https://github.com/anagntto) as part of a postgraduate program in Data Science.
