import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dash import Dash, dcc, html
import plotly.graph_objects as go

# 1️⃣ Load data
df = pd.read_csv("student_data.csv")

# Drop 'student_id' column
df.drop("student_id", axis=1, inplace=True)

# Encode gender (M=1, F=0)
df["gender"] = df["gender"].map({"M": 1, "F": 0})

# Separate features and target
X = df.drop("internship_success", axis=1)
y = df["internship_success"]

# Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-test split to ensure class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 2️⃣ Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# 3️⃣ Train models and gather results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })

# Feature importances from Random Forest
rf = models["Random Forest"]
feature_importances = rf.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# 4️⃣ Dash app setup
app = Dash(__name__)
app.title = "ML Model Performance Dashboard"

# 5️⃣ Layout
app.layout = html.Div([
    html.H1("📈 ML Model Performance Dashboard", style={"textAlign": "center"}),

    html.Div([
        dcc.Graph(
            figure={
                "data": [
                    go.Bar(
                        name=metric,
                        x=[r["Model"] for r in results],
                        y=[r[metric] for r in results]
                    )
                    for metric in ["Accuracy", "Precision", "Recall"]
                ],
                "layout": go.Layout(
                    title="Model Performance Metrics",
                    barmode="group",
                    yaxis=dict(title="Score"),
                    xaxis=dict(title="Model"),
                    height=500
                )
            }
        )
    ]),

    html.Div([
        html.H2("🔍 Random Forest Feature Importances"),
        dcc.Graph(
            figure={
                "data": [
                    go.Bar(
                        x=feature_df["Importance"],
                        y=feature_df["Feature"],
                        orientation="h",
                        marker=dict(color="teal")
                    )
                ],
                "layout": go.Layout(
                    title="Feature Importances",
                    xaxis=dict(title="Importance"),
                    yaxis=dict(title="Feature"),
                    height=500
                )
            }
        )
    ])
])

# 6️⃣ Run app
if __name__ == "__main__":
    app.run(debug=True)

