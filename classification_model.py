import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Cargar datos
df = pd.read_csv('data/hotel_bookings.csv')  # Ajusta la ruta

# Preprocesamiento
X = df[['lead_time', 'adults', 'required_car_parking_spaces', 'total_of_special_requests']]
y = df['is_canceled']

# Balancear clases
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Entrenamiento
model = RandomForestClassifier(
    max_depth=10,
    random_state=42,
    n_estimators=200,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluación
print(classification_report(y_test, model.predict(X_test)))

# Guardar modelo
joblib.dump(model, 'classification_model.pkl')
joblib.dump(smote, 'smote.pkl')