import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="🏋️ Workout Recommendation System", layout="wide")

st.title("🏋️ Workout Recommendation System")
st.markdown("""
This app recommends effective workouts based on your preferences like **Body Part**, **Equipment**, **Level**, and **Workout Type**.
""")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("your_workout_dataset.csv")
    df.dropna(inplace=True)
    df = df.drop(columns=["Unnamed: 0"])  # Remove index column
    df = df.rename(columns={
        "Title.1": "Desc",
        "Title.2": "Type"
    })
    return df

df = load_data()

# -------------------- 🤖 Confusion Matrix (Level prediction) --------------------
st.subheader("📊 Confusion Matrix: Predicting Workout Level")

# Encode data for classification
class_df = df.copy()
encoders = {}
for col in ['Type', 'BodyPart', 'Equipment', 'Level']:
    le = LabelEncoder()
    class_df[col] = le.fit_transform(class_df[col])
    encoders[col] = le

X = class_df[['Type', 'BodyPart', 'Equipment']]
y = class_df['Level']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Display confusion matrix
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoders['Level'].classes_)
disp.plot(ax=ax, cmap="Blues")
st.pyplot(fig)

# -------------------- 🔍 Workout Recommendation --------------------
st.sidebar.header("Select Your Preferences")

# Re-encode for KNN recommendation
knn_df = df.copy()
for col in ['Type', 'BodyPart', 'Equipment', 'Level']:
    knn_df[col] = encoders[col].transform(df[col])

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
features = ['Type', 'BodyPart', 'Equipment', 'Level']
knn.fit(knn_df[features])

# Sidebar inputs
type_input = st.sidebar.selectbox("Workout Type", df['Type'].unique())
body_input = st.sidebar.selectbox("Body Part", df['BodyPart'].unique())
equip_input = st.sidebar.selectbox("Equipment", df['Equipment'].unique())
level_input = st.sidebar.selectbox("Fitness Level", df['Level'].unique())

if st.sidebar.button("Get Workout Recommendations"):
    input_data = pd.DataFrame([[ 
        encoders['Type'].transform([type_input])[0],
        encoders['BodyPart'].transform([body_input])[0],
        encoders['Equipment'].transform([equip_input])[0],
        encoders['Level'].transform([level_input])[0]
    ]], columns=features)

    distances, indices = knn.kneighbors(input_data)

    st.subheader("🏆 Top 5 Recommended Workouts")
    for idx in indices[0]:
        workout = df.iloc[idx]
        with st.container():
            st.markdown(f"### 🏋️ {workout['Title']}")
            st.markdown(f"**Type:** {workout['Type']} | **Body Part:** {workout['BodyPart']} | **Equipment:** {workout['Equipment']} | **Level:** {workout['Level']}")
            st.markdown(f"**Description:** {workout['Desc']}")
            st.markdown("---")
