import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Load your dataset
df = pd.read_csv('car_data.csv')

# Step 2: Create 'car_age' column (assuming it's 2020)
df['car_age'] = 2020 - df['year']

# Step 3: Drop columns we don't need
df.drop(['name', 'year'], axis=1, inplace=True)

# Step 4: Encode categorical columns
for col in ['fuel', 'seller_type', 'transmission', 'owner']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Step 5: Features and Target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Step 6: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Save the model as model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
