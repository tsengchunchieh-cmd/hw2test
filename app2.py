import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Interactive Linear Regression", layout="centered")

st.title('Interactive Linear Regression Visualizer')

st.write("""
This app allows you to perform and visualize simple linear regression.
Upload your own CSV data, or use the generated sample data to explore.
""")

# --- Data Loading --- 
st.sidebar.header('1. Upload Your Data (Optional)')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV file uploaded successfully!")
else:
    st.sidebar.info("No CSV uploaded. Using generated sample data.")
    # Generate Sample Data if no file is uploaded
    np.random.seed(42)
    X_gen = 2 * np.random.rand(100, 1)
    y_gen = 4 + 3 * X_gen + np.random.randn(100, 1)
    data = pd.DataFrame({'Feature': X_gen.flatten(), 'Target': y_gen.flatten()})

st.subheader('Preview of Data')
st.dataframe(data.head())

# --- Feature Selection ---
st.sidebar.header('2. Select Features')

if data is not None:
    available_columns = data.columns.tolist()
    
    if len(available_columns) < 2:
        st.error("Please ensure your dataset has at least two columns for regression.")
    else:
        feature_column = st.sidebar.selectbox('Select X (Independent Variable)', available_columns)
        target_column = st.sidebar.selectbox('Select y (Dependent Variable)', [col for col in available_columns if col != feature_column])

        if feature_column and target_column:
            X = data[[feature_column]].values
            y = data[target_column].values

            # --- Model Training ---
            st.sidebar.header('3. Train Model')
            if st.sidebar.button('Run Linear Regression'):
                if X.shape[0] == 0 or y.shape[0] == 0:
                    st.error("Selected columns contain no data. Please check your data and selections.")
                else:
                    # Split data (optional, but good practice for evaluation)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                    st.subheader('Regression Model Results')
                    st.write(f"**Intercept:** {model.intercept_:.2f}")
                    st.write(f"**Coefficient (Slope):** {model.coef_[0]:.2f}")
                    st.write(f"**Regression Equation:** y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

                    r2_train = r2_score(y_train, y_pred_train)
                    r2_test = r2_score(y_test, y_pred_test)
                    st.write(f"**R-squared (Training Data):** {r2_train:.2f}")
                    st.write(f"**R-squared (Test Data):** {r2_test:.2f}")

                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write(f"**Mean Squared Error (Test Data):** {mse_test:.2f}")

                    # --- Visualization ---
                    st.subheader('Regression Plot')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=X.flatten(), y=y, ax=ax, label='Data Points', color='blue')
                    
                    # Plot the regression line over the full range of X
                    x_min, x_max = X.min(), X.max()
                    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
                    y_pred_range = model.predict(x_range)
                    ax.plot(x_range, y_pred_range, color='red', linestyle='-', linewidth=2, label='Regression Line')
                    
                    ax.set_xlabel(feature_column)
                    ax.set_ylabel(target_column)
                    ax.set_title(f'Linear Regression: {target_column} vs {feature_column}')
                    ax.legend()
                    st.pyplot(fig)

                    st.write("""
                    **How to interpret:**
                    *   The blue dots are your data points.
                    *   The red line is the best-fit linear regression line, which minimizes the sum of squared residuals.
                    *   The R-squared score indicates how well the model fits the data (0 to 1, higher is better).
                    *   Mean Squared Error (MSE) measures the average squared difference between the estimated values and the actual value.
                    """)
        else:
            st.warning("Please select both a feature and a target column.")
else:
    st.warning("Please upload a CSV file or use the generated sample data.")
