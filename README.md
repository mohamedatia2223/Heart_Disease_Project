# Heart Disease Prediction – Comprehensive ML Pipeline

##  Project Overview

This project implements a **full end-to-end machine learning pipeline** on the [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
It includes **data preprocessing, feature selection, PCA, supervised & unsupervised modeling, hyperparameter tuning, and deployment** using a **Streamlit web app + Ngrok** for public access.

The primary goal is to **analyze, predict, and visualize heart disease risks** while also exploring clustering patterns in the dataset.

---

## Objectives

* ✅ Clean & preprocess dataset (handle missing values, encoding, scaling).
* ✅ Apply **PCA** for dimensionality reduction.
* ✅ Perform **feature selection** (RFE, Chi-Square, Feature Importance).
* ✅ Train multiple supervised models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
* ✅ Explore unsupervised learning with **K-Means**
* ✅ Optimize models using **GridSearchCV / RandomizedSearchCV**.
* ✅ Deploy an interactive **Streamlit UI** for real-time predictions.
* ✅ Expose app publicly with **Ngrok**.

---

## Tools & Libraries

* **Python** 
* **Data Handling & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **ML & Feature Engineering:** Scikit-learn
* **Dimensionality Reduction:** PCA
* **Deployment:** Streamlit, Pyngrok

---

## Project Structure

```
Heart_Disease_Project/
│── data/
│   └── heart_disease.csv
│
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
│── models/
│   └── heart_disease_pipeline_svm.pkl
│
│── ui/
│   └── app.py   # Streamlit web app
│
│── deployment/
│   └── ngrok_setup.txt
│
│── results/
│   └── evaluation_metrics.txt

│── pipeline/
│   └── pipe.py

│── README.md
│── requirements.txt
│── .gitignore
```

---

## Running the Project

### 1️Setup Environment

```bash
git clone https://github.com/mohamedatia2223/Heart_Disease_Project.git
cd Heart_Disease_Project
pip install -r requirements.txt
```

### 2️⃣ Run pipe.py

For step-by-step analysis:

```bash
pipeline/pipe.py
```

### 3️⃣ Run Streamlit App

```bash
streamlit run ui/app.py
```

This launches the app locally at:
👉 [http://localhost:8501](http://localhost:8501)

### 4️⃣ Deploy with Ngrok (Optional)

```bash
ngrok authtoken <YOUR_TOKEN>
ngrok http 8501
```

You’ll get a **public link** to share your app.

---

## 📊 Results

* ✔ Cleaned dataset with selected features
* ✔ PCA-transformed dataset with variance plots
* ✔ Supervised models with evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
* ✔ K-Means & Hierarchical Clustering insights
* ✔ Optimized SVM pipeline saved as `.pkl`
* ✔ Interactive Streamlit UI for predictions
* ✔ Public app access via Ngrok

---

## 📘 Dataset

* **Source:** [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
* **Features:** Age, Sex, Chest Pain Type, Cholesterol, Max Heart Rate, etc.
* **Target:** Presence of heart disease

---

## 🤝 Contribution

Pull requests are welcome! If you’d like to extend this project (e.g., deep learning models, cloud deployment), feel free to fork and contribute.

---

## 📜 License

This project is licensed under the **MIT License**.

---

✨ With this pipeline, you get **data science + machine learning + deployment in one project** 🚀
