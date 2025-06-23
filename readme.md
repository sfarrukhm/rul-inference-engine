rul_inference/
│
├── model/                      # Stores model and preprocessing assets
│   ├── model.pt               # Trained PyTorch model
│   ├── minmax_params.json     # Scaler min/max values
│   └── drop_indices.txt       # Indices of columns to drop
│
├── app/                       # FastAPI service (inference backend)
│   ├── main.py                # FastAPI app entry point
│   ├── inference.py           # Core inference logic
│   ├── preprocessor.py        # Feature transformation logic
│   └── utils.py               # Helpers: file loading, smoothing, etc.
│
├── ui/                        # Streamlit UI (frontend)
│   ├── app.py                 # Streamlit app
│   └── api_client.py          # Sends data to FastAPI and gets RUL
│
├── requirements.txt
└── README.md
