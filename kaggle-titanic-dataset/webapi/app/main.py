from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from io import BytesIO
from app.model import Model
from sklearn.ensemble import GradientBoostingClassifier
from fastapi.responses import StreamingResponse

app = FastAPI()
model = Model(GradientBoostingClassifier())


async def csvToDataframe(file: UploadFile):
    try:
        contents = await file.read()
        return pd.read_csv(BytesIO(contents))
    except:
        raise HTTPException(
            status_code=400,
            detail=f"failed to read uploaded file {file.filename}- should be a csv file")


@app.post("/train")
async def train(file: UploadFile = File(...)):
    train_df = await csvToDataframe(file)
    model.fit(train_df)
    return {"trained": "success"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="cannot predict because the model is not trained, use /train service first")

    test_df = await csvToDataframe(file)
    predictions = model.predict(test_df)
    predict_df = pd.DataFrame({"PassengerId": test_df.PassengerId, "Survived": predictions})
    stream = BytesIO()

    predict_df.to_csv(stream, index=False)

    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv",
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response
