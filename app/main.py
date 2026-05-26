import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from inference import load_artifacts, predict, build_grid

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(title="KL House Price Predictor")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/static/{filepath:path}")
async def static_files(filepath: str):
    file = BASE_DIR / "static" / filepath
    if not file.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    ct = {".css": "text/css", ".js": "application/javascript"}.get(file.suffix, "application/octet-stream")
    return Response(content=file.read_bytes(), media_type=ct,
                    headers={"Cache-Control": "no-store"})


print("Loading model and preprocessors...")
model, preprocessors = load_artifacts(MODEL_DIR)
print("Building prediction grid...")
GRID = build_grid(model, preprocessors)
print(f"Ready — grid has {len(GRID):,} entries")


class PredictRequest(BaseModel):
    property_type: str
    district: str
    mukim: str
    scheme_name: str = ""
    road_name: str = ""
    tenure: str
    transaction_month: int
    transaction_year: int
    land_parcel_area: float
    unit_level: int


class SearchRequest(BaseModel):
    budget_min: float
    budget_max: float
    tenure: str = ""
    property_type: str = ""
    mukim: str = ""
    min_area: float = 0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    try:
        price = predict(req.model_dump(), model, preprocessors)
        return JSONResponse({"price": price, "formatted": f"RM {price:,.0f}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    results = [
        r for r in GRID
        if req.budget_min <= r["predicted_price"] <= req.budget_max
        and (not req.tenure or r["tenure"] == req.tenure)
        and (not req.property_type or r["property_type"] == req.property_type)
        and (not req.mukim or r["mukim"] == req.mukim)
        and r["land_parcel_area"] >= req.min_area
    ]
    results.sort(key=lambda x: x["land_parcel_area"], reverse=True)
    for r in results:
        r["formatted_price"] = f"RM {r['predicted_price']:,}"
    return JSONResponse({"results": results[:40], "count": len(results)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
