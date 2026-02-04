from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import io
import time

app = FastAPI()
templates = Jinja2Templates(directory="./templates")

# Load ML model
with open("models/latest_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------- HOME ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ---------- REAL-TIME PAGE ----------
@app.get("/realtime", response_class=HTMLResponse)
def realtime_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "page": "realtime"}
    )

# ---------- REAL-TIME PREDICT ----------
@app.post("/predict", response_class=HTMLResponse)
def realtime_predict(
    request: Request,
    overs: float = Form(...),
    runs: int = Form(...),
    wickets: int = Form(...),
    runs_last_5: int = Form(...),
    wickets_last_5: int = Form(...)
):
    X = [[overs, runs, wickets, runs_last_5, wickets_last_5]]
    prediction = int(model.predict(X)[0])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "page": "realtime",
            "realtime_prediction": prediction
        }
    )

# ---------- BATCH PAGE ----------
@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "page": "batch"}
    )

# ---------- BATCH PREDICT ----------
@app.post("/batch-predict", response_class=HTMLResponse)
async def batch_predict(
    request: Request,
    file: UploadFile = File(...)
):
    time.sleep(3)  # simulate batch delay

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    cols = ["overs", "runs", "wickets", "runs_last_5", "wickets_last_5"]
    df["predicted_final_score"] = model.predict(df[cols])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "page": "batch",
            "batch_result": df.to_html(index=False)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
