# 🏠 Home Rent Prediction Dashboard
> A machine learning web application that uses **Simple Linear Regression** to predict property rental prices based on square footage.

---

## 🔗 Live Demo
Test the prediction model here:  
** **

---

## ✨ Features
* **Real-time Prediction:** Enter any area (sq ft) in the sidebar to get an instant rental estimate.
* **Interactive Visualization:** View a scatter plot of actual data points overlaid with the calculated **Linear Regression Line**.
* **Model Statistics:** Displays the model's coefficients ($m$) and intercept ($b$) using the linear equation $y = mx + b$.
* **Data Transparency:** View the raw dataset used to train the model directly in the app.

---

## 🛠️ Technical Implementation
The project implements a **Supervised Learning** workflow:
1.  **Data Processing:** Utilizing `Pandas` to handle the `home rent (1).csv` dataset.
2.  **Model Training:** Training a `LinearRegression()` model from `scikit-learn`.
3.  **Optimization:** The model minimizes the Mean Squared Error (MSE) to find the best-fit line.

---

## 📂 Project Structure
```text
├── app.py                # Streamlit app with ML logic
├── requirements.txt      # Required libraries (sklearn, pandas, etc.)
├── home rent (1).csv     # Training dataset
└── README.md             # Project documentation
