import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load dataset for analytics
df = pd.read_csv('car_data.csv')
df['car_age'] = 2020 - df['year']

# Encode for analytics view (same as training)
encode_dicts = {
    'fuel': {'CNG': 0, 'Diesel': 1, 'Petrol': 2},
    'seller_type': {'Dealer': 0, 'Individual': 1},
    'transmission': {'Automatic': 0, 'Manual': 1},
    'owner': {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3}
}
for col, mapping in encode_dicts.items():
    df[col] = df[col].map(mapping)

# Streamlit app layout
st.set_page_config(page_title="Car Price App", layout="wide")
st.sidebar.title("🔎 Navigation")
selection = st.sidebar.radio("Go to", ["🏠 Predictor", "📊 Analytics"])

# ---------------------------
# 1. Car Price Predictor Page
# ---------------------------
if selection == "🏠 Predictor":
    st.title("🚗 Car Price Predictor")
    st.markdown("Enter car details to estimate the selling price.")

    # --- INPUTS ---
    car_age = st.slider("Car Age (in years)", 0, 20, 5)
    km_driven = st.number_input("Kilometers Driven", 0, 500000, step=1000, value=30000)
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
    transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
    owner = st.selectbox("Number of Previous Owners", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    # --- ENCODING ---
    fuel_encoded = encode_dicts['fuel'][fuel_type]
    seller_encoded = encode_dicts['seller_type'][seller_type]
    trans_encoded = encode_dicts['transmission'][transmission]
    owner_encoded = encode_dicts['owner'][owner]

    if st.button("Predict Price"):
        input_data = np.array([[km_driven, fuel_encoded, seller_encoded, trans_encoded, owner_encoded, car_age]])
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {round(predicted_price, 2)}")

# ------------------------
# 2. Analytics Page
# ------------------------
elif selection == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    st.markdown("Explore the dataset with visual insights.")

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig1, ax1 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Selling Price vs Kilometers Driven")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="km_driven", y="selling_price", hue="fuel", palette="viridis", ax=ax2)
    ax2.set_title("Selling Price vs KM Driven")
    st.pyplot(fig2)

    st.subheader("Average Selling Price by Fuel Type")
    avg_price_fuel = df.groupby('fuel')['selling_price'].mean().reset_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(x="fuel", y="selling_price", data=avg_price_fuel, ax=ax3)
    ax3.set_xlabel("Fuel Type (0=CNG, 1=Diesel, 2=Petrol)")
    ax3.set_ylabel("Average Selling Price")
    ax3.set_title("Avg Price by Fuel Type")
    st.pyplot(fig3)

    st.subheader("Distribution of Car Age")
    fig4, ax4 = plt.subplots()
    sns.histplot(df['car_age'], kde=True, bins=20, ax=ax4)
    ax4.set_title("Distribution of Car Age")
    st.pyplot(fig4)
