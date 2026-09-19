import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("demand_forecasting_model.pkl")

# Page config
st.set_page_config(page_title="Store Sales Prediction", layout="wide")

# Title
st.title("📊 Store Sales Prediction Dashboard")
st.write("Predict store sales based on store conditions.")

# Layout
col1, col2 = st.columns(2)

# Column 1 (Store Info)
with col1:
    st.subheader("Store Information")

    # Day of Week
    days_map = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7
    }

    selected_day = st.selectbox("Day Of Week", list(days_map.keys()))
    DayOfWeek = days_map[selected_day]

    # Customers
    Customers = st.number_input(
        "Number of Customers", 
        min_value=0, 
        max_value=50000, 
        value=500
    )

    # School Holiday
    SchoolHoliday_dict = {
        "True": 1,
        "False": 0  
    }

    selected_SchoolHoliday = st.selectbox(
        "School Holiday", 
        list(SchoolHoliday_dict.keys())
    )
    SchoolHoliday = SchoolHoliday_dict[selected_SchoolHoliday]

    # Year
    Year = st.selectbox("Year", [2013, 2014, 2015])


# Column 2 (Promo Info)
with col2:
    st.subheader("Promotion Information")

    # Month
    Month = st.selectbox("Month", list(range(1, 13)))

    # Promo
    is_promo_dict = {
        "True": 1,
        "False": 0
    }

    selected_is_promo = st.selectbox(
        "Is Promo", 
        list(is_promo_dict.keys())
    )
    IsPromo = is_promo_dict[selected_is_promo]

    # Promo2
    selected_is_promo2 = st.selectbox(
        "Is Promo2", 
        list(is_promo_dict.keys())
    )
    IsPromo2 = is_promo_dict[selected_is_promo2]

    # Competition Duration
    CompetitionDurationMonths = st.number_input(
        "Competition Duration in Months", 
        min_value=0.0, 
        value=12.0
    )
    CompetitionDurationMonths_log = np.log1p(CompetitionDurationMonths)


st.write("")


# Prediction
if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "DayOfWeek": [DayOfWeek],
        "Customers": [np.log1p(Customers)],
        "SchoolHoliday": [SchoolHoliday],
        "Year": [Year],
        "Month": [Month],
        "IsPromo": [IsPromo],
        "IsPromo2": [IsPromo2],
        "CompetitionDurationMonths_log": [CompetitionDurationMonths_log]
    })

    st.write("Input Data:", input_data)

    # Prediction
    prediction = model.predict(input_data)
    prediction_value = np.expm1(prediction)[0]

    # 🎯 عرض النتيجة بشكل احترافي
    st.markdown(f"""
    ### 💰 Predicted Sales  
    # ${prediction_value:,.2f}
    """)

    # 🧠 حساب متوسط البيع لكل عميل
    if Customers > 0:
        avg_sale = prediction_value / Customers

        st.info(f"🧍 Average Sale per Customer: ${avg_sale:,.2f}")

        # تقييم منطقي
        if avg_sale < 5:
            st.warning("⚠️ Low sales per customer")
        elif avg_sale <= 20:
            st.success("✅ Normal sales range")
        else:
            st.error("🚨 Unusually high sales per customer")

    # 🔥 Insights ذكية
    st.subheader("📊 Insights")

    if IsPromo == 1:
        st.write("📈 Promo is active → expected increase in sales")
    else:
        st.write("📉 No promo → sales might be lower")
        st.write("💡 Tip: Activating a promo could increase sales")

    if Customers < 200:
        st.write("👥 Low customer traffic → lower expected sales")
    elif Customers > 1000:
        st.write("🔥 High customer traffic → strong sales expected")

    if CompetitionDurationMonths < 6:
        st.write("🏪 New competition nearby → possible negative impact")
    else:
        st.write("🏪 Competition stabilized → less impact on sales")
