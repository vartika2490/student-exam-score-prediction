# Student Exam Score Prediction

This project predicts a student's final exam score using regression techniques based on features such as:
- Hours studied per week
- Attendance rate
- Previous exam scores

It helps educators understand how key academic behaviors affect outcomes and potentially identify students who may need academic support.

---

## Dataset Description
The dataset is extracted from a ZIP file located at:
```
C:/Users/Acer/Downloads/student_performance_dataset.zip
```
It contains the following features:
- `Study_Hours_per_Week`
- `Attendance_Rate`
- `Past_Exam_Scores`
- `Final_Exam_Score` (Target)
- `Gender`, `Internet_Access_at_Home`, `Extracurricular_Activities` (used for encoding if needed)

---

## ML Approach
We use a Linear Regression model from `scikit-learn`:
- Split data into train/test sets (80/20)
- Fit a regression model
- Evaluate using:
  - Mean Squared Error (MSE)
  - R-squared (R²)
- Visualize results with Seaborn

---

## Requirements
Install dependencies by running:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- matplotlib
- seaborn
- scikit-learn

---

## Running the Project
```bash
python main.py
```
- Automatically extracts the dataset from ZIP if not already done
- Trains the model and displays evaluation results
- Shows a styled scatter plot comparing actual and predicted scores

---

## Output Example
- Mean Squared Error (MSE): 15.78
- R-squared (R²): 0.84

![Model Visualization](#) *(Save plot as image to replace this placeholder)*

---

## Insights
- Students who study more, attend regularly, and perform well in past exams tend to have higher final scores
- Visualization helps identify outliers and evaluate prediction accuracy

---

## Future Enhancements
- Add Random Forest Regressor or other models
- Save trained model as `.pkl`
- Deploy model with Flask or Streamlit
- Include more features (e.g., sleep hours, motivation score)

---

## Project Structure
```
StudentExamPrediction/
├── main.py
├── requirements.txt
├── README.md
└── extracted_data/
    └── student_performance_dataset.csv
```

---

Feel free to contribute or fork this project for your own educational or research use!
