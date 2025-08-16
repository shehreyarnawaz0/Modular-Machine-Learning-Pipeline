# 🚢 Titanic Survival – Modular ML Pipeline

A clean, reusable ML pipeline in **Google Colab** / Python:
- **Preprocessing:** missing values, encoding, scaling, train/test split  
- **Training:** Logistic Regression, Decision Tree, Random Forest with **GridSearchCV**  
- **Evaluation:** Accuracy, Precision, Recall, F1, classification reports, confusion matrices, feature importance  

---

## 📊 Dataset
- Source: Titanic CSV (public mirror)  
- Download in code:  
  `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`  
- Target: `Survived` (0/1)  

---

## ⚙️ Run
1. Clone this repo  
   ```bash
   git clone https://github.com/yourusername/titanic-ml-pipeline.git
   cd titanic-ml-pipeline
   ```
2. Install requirements  
   ```bash
   pip install -r requirements.txt
   ```
3. Run training + evaluation  
   ```bash
   python src/train.py
   ```

---

## 🧠 Models
- Logistic Regression  
- Decision Tree  
- Random Forest  

Best results typically from **Random Forest (~0.84 accuracy)**  

---

## 📦 Requirements
`pandas, numpy, scikit-learn, matplotlib, seaborn, joblib`  

---

## 📁 Structure
```
.
├── data/                 
├── models/               
├── src/                  
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── notebook/
│   └── titanic_pipeline.ipynb   # optional
├── README.md
└── requirements.txt
```

---

## 📝 License
MIT
