import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

# Cargar datos
df = pd.read_csv('data/airbnb.csv')  # Ajusta la ruta según tu estructura

# Preprocesamiento
X = df[['neighbourhood', 'room_type', 'minimum_nights', 'number_of_reviews']]
y = df['price']

# One-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['neighbourhood', 'room_type']])
X_num = X[['minimum_nights', 'number_of_reviews']].values
X_processed = pd.concat([pd.DataFrame(X_encoded.toarray()), pd.DataFrame(X_num)], axis=1)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Entrenamiento
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# Evaluación
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {mse**0.5:.2f}")

# Guardar modelo
joblib.dump(model, 'regression_model.pkl')
joblib.dump(encoder, 'encoder.pkl')