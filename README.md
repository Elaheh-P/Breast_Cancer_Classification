# 🩺 Breast Cancer Classification - Machine Learning Project

## 📌 Introduction
Breast cancer is one of the most common cancers worldwide. Early and accurate diagnosis can significantly improve patient outcomes. This project applies **seven different machine learning models** to classify breast cancer using the **Breast Cancer Wisconsin Dataset** from `sklearn.datasets.load_breast_cancer`.

The goal of this project is to compare multiple classification models and analyze their performance based on different evaluation metrics.

## 📂 What Was Done
### **1️⃣ Data Preprocessing**
- Loaded the dataset and examined feature distributions.
- Split the data into **training (80%)** and **testing (20%)** sets.
- Applied **MinMaxScaler** to normalize feature values between 0 and 1.

### **2️⃣ Model Training**
We trained and compared the following **7 machine learning models**:
- **Naive Bayes**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Artificial Neural Network (ANN)**

Each model was trained using **Scikit-learn**, with carefully tuned hyperparameters.

### **3️⃣ Model Evaluation**
The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Then these metrics were stored in a **Pandas DataFrame** for easy comparison and visualization.

### **4️⃣ Results & Visualization**
We plotted bar charts to visualize and compare each model's:
- **Test Accuracy**
- **Train Accuracy**
- **Recall**
- **Precision**
- **F1-Score**

## 📊 Model Performance Results

| Model               | Train Accuracy | Test Accuracy | Recall  | Precision | F1-Score |
|---------------------|---------------|--------------|---------|-----------|----------|
| Naive Bayes        | 0.936264      | 0.964912     | 0.985915 | 0.958904  | 0.972222 |
| KNN                | 0.975824      | 0.964912     | 0.971831 | 0.971831  | 0.971831 |
| Decision Tree      | 0.980220      | 0.964912     | 1.000000 | 0.946667  | 0.972603 |
| Random Forest      | 0.995604      | 0.964912     | 0.985915 | 0.958904  | 0.972222 |
| SVM                | 0.989011      | 0.982456     | 1.000000 | 0.972603  | 0.986111 |
| Logistic Regression| 0.969231      | 0.982456     | 1.000000 | 0.972603  | 0.986111 |
| ANN                | 0.993407      | 0.982456     | 1.000000 | 0.972603  | 0.986111 |



> 🚀 **Key Insights:**
- The best-performing models in terms of accuracy were "Random Forest" and "ANN", with an accuracy of  0.995604  and  0.993407 respectively.
- The highest recall (equal to 1) was achieved by "Decision Tree" , "SVM", "Logistic Regression" , and "ANN" making them the best choice for detecting cancer cases.
- The highest precision was seen in "SVM", "Logistic Regression" , and "ANN", meaning they make fewer false-positive predictions.
- The best F1-score was achieved by "SVM", "Logistic Regression" , and "ANN", suggesting a balanced approach between recall and precision.

**Final Conclusion:**

The models SVM, Logistic Regression, and ANN performed equally well in terms of both Precision and Recall.
Since these two factors are critical for medical diagnosis, we can confidently choose any of these three models depending on computational efficiency or other constraints.
If the goal is real-time predictions, Logistic Regression may be preferable due to its simplicity and speed.
If we need a more flexible decision boundary, SVM or ANN could be better choices.

## ✉ Contact
If you have any questions or feedback, feel free to reach out!
- **LinkedIn:** 

🚀 **If you find this project useful, please ⭐ star the repository!** ⭐

