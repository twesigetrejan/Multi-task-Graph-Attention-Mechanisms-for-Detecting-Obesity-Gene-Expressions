import bentoml
import pandas as pd
import pickle
import os  # Add this import
from bentoml.io import JSON
from pydantic import BaseModel

# Load the model from BentoML
rf_model = bentoml.sklearn.get("obesity_risk_model:latest").to_runner()

# Get the directory of the current script
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load encoders using relative paths
with open(os.path.join(SAVE_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(SAVE_DIR, "target_encoder.pkl"), "rb") as f:
    target_le = pickle.load(f)

with open(os.path.join(SAVE_DIR, "meals.pkl"), "rb") as f:
    meal_df = pickle.load(f)

# Define the expected input format
class UserProfile(BaseModel):
    Age: int
    BMI: float
    Physical_Activity: str
    Diet_Type: str
    MC4R_Present: int
    MC4R_Variant: str
    PPARG_Present: int
    PPARG_Variant: str
    FTO_Present: int
    FTO_Variant: str
    LEPR_Present: int
    LEPR_Variant: str

# Define the BentoML service
svc = bentoml.Service("obesity_risk_service", runners=[rf_model])

# Define the prediction endpoint
@svc.api(input=JSON(), output=JSON())
def predict(user: dict):  # Accept raw JSON as a dict
    # Convert the input dict to a UserProfile instance
    user_profile = UserProfile(**user)

    # Convert user input into DataFrame
    user_df = pd.DataFrame([user_profile.dict()])

    # Encode categorical features
    for col in ["Diet_Type", "Physical_Activity"]:
        user_df[col] = label_encoders[col].transform(user_df[col])

    for col in ["MC4R_Variant", "PPARG_Variant", "FTO_Variant", "LEPR_Variant"]:
        user_df[col] = label_encoders[col].transform(user_df[col].astype(str))

    # Define feature order
    features = [
        "Age", "BMI", "Physical_Activity", "Diet_Type",
        "MC4R_Present", "MC4R_Variant",
        "PPARG_Present", "PPARG_Variant",
        "FTO_Present", "FTO_Variant",
        "LEPR_Present", "LEPR_Variant"
    ]
    user_df = user_df[features]

    # Predict obesity risk
    prediction = rf_model.predict.run(user_df)
    predicted_label = target_le.inverse_transform(prediction)[0]

    # Meal recommendation logic
    def recommend_meals(predicted_category, meal_df, num_meals=5):
        if predicted_category == 'Low':
            preferred_clusters = [0, 1, 2, 3]
        elif predicted_category == 'Medium':
            preferred_clusters = [4, 5, 6, 7]
        else:
            preferred_clusters = [8, 9]

        recommended_meals = meal_df[meal_df['Meal_Cluster'].isin(preferred_clusters)]
        recommended_meals = recommended_meals.sample(frac=1).reset_index(drop=True)
        
        return recommended_meals[['Descrip', 'Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g']].head(num_meals).to_dict(orient="records")

    # Get 5 recommended meals
    recommended_meals = recommend_meals(predicted_label, meal_df, num_meals=5)

    return {
        "predicted_obesity_risk": predicted_label,
        "recommended_meals": recommended_meals
    }