#  RUL Inference Engine

A microservice-based system for predicting the **Remaining Useful Life (RUL)** of aircraft engines using sensor time-series data from the NASA C-MAPSS dataset. Built with **FastAPI** (inference backend) and **Streamlit** (interactive UI).

---
## 🚀 Running the Project

**Start the inference service:**

```bash
cd app/
uvicorn main:app --reload
```
**Launch the UI:**
```
cd UI/
streamlit run main.py
```
### 🧪 Example Prediction Flow

1. User enters last 20 cycles of sensor data in the Streamlit UI.  
2. UI sends a POST request to the FastAPI inference service.  
3. FastAPI service performs the following steps:
   - Applies preprocessing (drop, smooth, scale)
   - Runs the LSTM model to predict RUL
   - Returns the predicted RUL value  
4. UI displays the predicted RUL in real time.

---

### 📁 Artifacts

| File                  | Purpose                          |
|-----------------------|----------------------------------|
| `checkpoint.pt`            | Trained LSTM weights             |
| `minmax_params.json`  | Feature scaling parameters       |

### 📚 Dataset Source

This project is trained on the NASA C-MAPSS dataset available on Kaggle:  
[🔗 NASA C-MAPSS: Turbofan Engine Degradation Simulation Data](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data)


