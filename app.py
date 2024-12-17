import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Airline Satisfaction Prediction Model", page_icon="✈")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!", "About Me"])

# Load dataset
df = pd.read_csv('data/train_cleaned.csv')

# Define numerical columns globally
num_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()

# Preprocessing the data (Define outside of page-specific code)
X = df.drop(['satisfaction'], axis=1)
y = df['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training models globally
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

# Home Page
if page == "Home":
    st.title("Airline Customer Satisfaction Prediction Model")
    st.write("""
        This app provides an interactive platform to explore the airline customer satisfaction.
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        **Use the sidebar to navigate through the sections.**

        Which features or variables do you think make for a satisfying airline experience?

        Dig into our model to see if your predictions were correct!
    """)
    ## st.write("👈 Use the sidebar to navigate between different sections.")
    st.image('images/satisfied_and_un.jpeg', caption="Will Gender Make a Difference in Satisfaction with the Airline?")

# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")
    st.subheader("About the Data")
    st.write("""
        Airlines are constantly working to improve customer satisfaction. 
        But what really makes customers happy during their flight experience? 
        
        This dataset features over 100,000 rows and 23 variables to help answer this question.  
        Whether it's inflight entertainment, seat comfort, or even gate location, there are 
        insights waiting to be uncovered. Let your data detective mind take flight?
    """)

    # Data Dictionary Section (moved first)
    if st.checkbox("Show Data Dictionary 🧾"):
        st.subheader("Data Dictionary")
        st.write("""
        - `Age` (int): Age of the customer.  
        - `Flight Distance` (int): Distance of the flight in miles.  
        - `Inflight WiFi Service` (int): Rating of inflight Wi-Fi service (0 to 5).  
        - `Departure/Arrival Time Convenience` (int): Convenience of departure/arrival time (0 to 5).  
        - `Ease of Online Booking` (int): Ease of booking online (0 to 5).  
        - `Gate Location` (int): Convenience of gate location (0 to 5).  
        - `Food and Drink` (int): Rating of inflight food and drink (0 to 5).  
        - `Online Boarding` (int): Rating of online boarding process (0 to 5).  
        - `Seat Comfort` (int): Comfort of seat (0 to 5).  
        - `Inflight Entertainment` (int): Rating of inflight entertainment (0 to 5).  
        - `On-board Service` (int): Rating of onboard service (0 to 5).  
        - `Leg Room Service` (int): Satisfaction with legroom (0 to 5).  
        - `Baggage Handling` (int): Satisfaction with baggage handling (0 to 5).  
        - `Check-in Service` (int): Satisfaction with check-in service (0 to 5).  
        - `Inflight Service` (int): Satisfaction with inflight service (0 to 5).  
        - `Cleanliness` (int): Cleanliness of the aircraft (0 to 5).  
        - `Departure Delay in Minutes` (int): Minutes of delay at departure.  
        - `Arrival Delay in Minutes` (float): Minutes of delay at arrival.  
        - `Gender (Male)` (int): Whether the customer is male (1 for male, 0 for female).  
        - `Customer Type: Disloyal` (int): Whether the customer is disloyal (1 for disloyal, 0 for loyal).  
        - `Travel Type: Personal` (int): Whether the travel is personal (1 for personal, 0 for business).  
        - `Class: Economy` (int): Whether the customer is in economy class (1 for yes, 0 for no).  
        - `Class: Economy Plus` (int): Whether the customer is in economy plus class (1 for yes, 0 for no).  
        - `Satisfaction` (int): Customer satisfaction (0 for unsatisfied, 1 for satisfied).  ⭐ *The Target Variable*
        """)
        st.write("**Description**: This dataset provides key metrics for analyzing airline customer satisfaction.")

    # Dataset Display
    if st.checkbox("Show DataFrame 🔎"):
        st.subheader("Quick Glance at the Data")
        st.dataframe(df)

    # Shape of Dataset
    if st.checkbox("Show Shape of Data 📏"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Image with caption
    st.image(
        'images/gate3.jpeg', 
        caption="Is he satisfied? One thing's for sure... He likes his suitcase."
    )

# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    # Select the type of visualization options before the image
    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Satisfaction Correlation'])

    # Histograms
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Satisfaction", value=True):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    # Box Plots
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='satisfaction', y=b_selected_col, title=chart_title, color='satisfaction'))

    # Satisfaction Correlation (Top Positive & Negative Correlations)
    if 'Satisfaction Correlation' in eda_type:
        st.subheader("Satisfaction Correlation - Visualizing Satisfaction vs Other Numerical Variables")
        if 'satisfaction' in num_cols:
            satisfaction_corr = df[num_cols].corrwith(df['satisfaction'])
            satisfaction_corr_sorted = satisfaction_corr.sort_values(ascending=False)
            st.bar_chart(satisfaction_corr_sorted)

    # Move the image below all visualizations
    st.image(
        'images/dubai.jpeg', 
        caption="Customers with longer flights tended to rate themselves as more satisfied, compared to the shorter flights.\nSee what other fun insights you can find."
    )

# Model Training and Evaluation
elif page == "Model Training and Evaluation":
    st.title("🤖 Model Training and Evaluation")
    st.subheader("Train Multiple Models and Evaluate Performance")
    st.write("There are over 100,000 rows, so the models load slowly, be patient.")

    # Model selection
    model_option = st.selectbox("Select the model to train:", models.keys())

    # Train model
    model = models[model_option]
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)

    # Confusion Matrix Display
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax)
    st.pyplot(fig)

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Extract False Positives (FP) and False Negatives (FN)
    tn, fp, fn, tp = cm.ravel()
    
    # Display evaluation metrics
    st.subheader("Model Evaluation Metrics")
    accuracy = model.score(X_test_scaled, y_test)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    
    # False Positives and False Negatives
    st.write(f"False Positives (FP): {fp}")
    st.write(f"False Negatives (FN): {fn}")

    # Explanation of False Positives and False Negatives
    st.write("""
    - **False Positives (FP)**: These are instances where the model incorrectly predicted a positive outcome (e.g., predicted 'satisfied' when the actual label was 'unsatisfied').
    - **False Negatives (FN)**: These are instances where the model incorrectly predicted a negative outcome (e.g., predicted 'unsatisfied' when the actual label was 'satisfied').
    
    In other words:
    - A **false positive** happens when a customer who was actually unsatisfied is incorrectly predicted as satisfied.
    - A **false negative** happens when a customer who was actually satisfied is incorrectly predicted as unsatisfied.
    """)



# Make Predictions Page
elif page == "Make Predictions!":
    st.title("✈ Make Predictions on Airline Satisfaction")

    st.subheader("Adjust the values below to make predictions on customer satisfaction:")

    # User inputs for prediction
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    flight_distance = st.slider("Flight Distance (miles)", min_value=50, max_value=5000, value=1000)
    inflight_wifi_service = st.slider("Inflight WiFi Service", min_value=0, max_value=5, value=3)
    departure_arrival_time_convenience = st.slider("Departure/Arrival Time Convenience", min_value=0, max_value=5, value=3)
    ease_of_online_booking = st.slider("Ease of Online Booking", min_value=0, max_value=5, value=3)
    gate_location = st.slider("Gate Location", min_value=0, max_value=5, value=3)
    food_and_drink = st.slider("Food and Drink", min_value=0, max_value=5, value=3)
    online_boarding = st.slider("Online Boarding", min_value=0, max_value=5, value=3)
    seat_comfort = st.slider("Seat Comfort", min_value=0, max_value=5, value=3)
    inflight_entertainment = st.slider("Inflight Entertainment", min_value=0, max_value=5, value=3)
    on_board_service = st.slider("On-board Service", min_value=0, max_value=5, value=3)
    leg_room_service = st.slider("Leg Room Service", min_value=0, max_value=5, value=3)
    baggage_handling = st.slider("Baggage Handling", min_value=0, max_value=5, value=3)
    checkin_service = st.slider("Check-in Service", min_value=0, max_value=5, value=3)
    inflight_service = st.slider("Inflight Service", min_value=0, max_value=5, value=3)
    cleanliness = st.slider("Cleanliness", min_value=0, max_value=5, value=3)
    departure_delay_minutes = st.slider("Departure Delay (minutes)", min_value=0, max_value=1000, value=0)
    arrival_delay_minutes = st.slider("Arrival Delay (minutes)", min_value=0, max_value=1000, value=0)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    customer_type = st.selectbox("Customer Type", ['Loyal', 'Disloyal'])
    travel_type = st.selectbox("Travel Type", ['Business', 'Personal'])
    flight_class = st.selectbox("Class", ['Economy', 'Economy Plus'])

    # Prepare user input data
    user_input = pd.DataFrame({
        'age': [age],
        'flight_distance': [flight_distance],
        'inflight_wifi_service': [inflight_wifi_service],
        'departure/arrival_time_convenient': [departure_arrival_time_convenience],
        'ease_of_online_booking': [ease_of_online_booking],
        'gate_location': [gate_location],
        'food_and_drink': [food_and_drink],
        'online_boarding': [online_boarding],
        'seat_comfort': [seat_comfort],
        'inflight_entertainment': [inflight_entertainment],
        'on-board_service': [on_board_service],
        'leg_room_service': [leg_room_service],
        'baggage_handling': [baggage_handling],
        'checkin_service': [checkin_service],
        'inflight_service': [inflight_service],
        'cleanliness': [cleanliness],
        'departure_delay_in_minutes': [departure_delay_minutes],
        'arrival_delay_in_minutes': [arrival_delay_minutes],
        'gender_Male': [1 if gender == 'Male' else 0],
        'customer_type_disloyal Customer': [1 if customer_type == 'Disloyal' else 0],
        'type_of_travel_Personal Travel': [1 if travel_type == 'Personal' else 0],
        'class_Eco': [1 if flight_class == 'Economy' else 0],
        'class_Eco Plus': [1 if flight_class == 'Economy Plus' else 0]
    })

    # Ensure the columns in user_input match the training data
    expected_columns = [
        'age', 'flight_distance', 'inflight_wifi_service', 'departure/arrival_time_convenient', 
        'ease_of_online_booking', 'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort', 
        'inflight_entertainment', 'on-board_service', 'leg_room_service', 'baggage_handling', 'checkin_service', 
        'inflight_service', 'cleanliness', 'departure_delay_in_minutes', 'arrival_delay_in_minutes', 'gender_Male', 
        'customer_type_disloyal Customer', 'type_of_travel_Personal Travel', 'class_Eco', 'class_Eco Plus'
    ]
    user_input = user_input[expected_columns]

    st.write("### Your Input Values")
    st.dataframe(user_input)

    # Preprocessing
    model = KNeighborsClassifier(n_neighbors=9)
    X = df.drop(columns='satisfaction')  # Drop target column
    y = df['satisfaction']

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the dataset
    model.fit(X_scaled, y)

    # Make prediction
    prediction = model.predict(user_input_scaled)[0]

    # Display the prediction result
    st.write(f"The model predicts that the customer will be: **{'Satisfied 👍' if prediction == 1 else 'Unsatisfied 👎'}**")

# About Me
elif page == "About Me":
    st.title("About Me")
    st.write("""
    ## Hi, I'm Mike Strusz! 👋

    I'm a Data Analyst based in **Milwaukee**, passionate about solving real-world problems through data-driven insights. I have a strong background in data analysis, data visualization, and machine learning, and I’m continuously expanding my skills to stay at the forefront of the field.

    I believe in the power of collaboration, and I’m always open to discussing ideas, sharing knowledge, or exploring new opportunities.

    I was a teacher for over a decade, and during that time, I developed a passion for making learning engaging and accessible. Transitioning into data analysis has been a natural next step for me, combining my love for problem-solving and working with data. 

    This project was particularly enjoyable because I love to travel, and thinking about what predicts a satisfying trip was fun for me. Exploring these patterns gives me new insights into the factors that contribute to positive experiences.

    ### Let’s Connect!
    Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/mike-strusz) to discuss this topic further or explore potential collaborations. I'd love to hear from you!

    📧 You can also reach me via email at [mike.strusz@gmail.com](mailto:mike.strusz@gmail.com).
    """)

    # Insert the image
    # Insert the image
    st.image("images/mike.jpeg", caption="Me on the Milwaukee Riverwalk, wearing one of my 50+ bowties.", use_container_width=True)
