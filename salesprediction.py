import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Advertising.csv")  # ← Replace with your actual file name/path

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Predicting new data
new_data = pd.DataFrame({'TV': [150], 'Radio': [25], 'Newspaper': [20]})
prediction = model.predict(new_data)
print(f"Predicted Sales: {prediction[0]:.2f}")
