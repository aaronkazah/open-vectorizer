# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import tempfile

from starlette.responses import FileResponse

from vectorizer import Vectorizer  # Assuming the Vectorizer class is in vectorizer.py

app = FastAPI()

# Mount static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the Vectorizer
model_path = "./assets/sam_vit_l_0b3195.pth"  # Update with your SAM model path
vectorizer = Vectorizer(model_path)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    temp_file_path = f"./temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_file = "output.svg"
    vectorizer.vectorize(image_path=temp_file_path, output_file=output_file)

    os.remove(temp_file_path)
    return FileResponse(output_file, media_type="image/svg+xml", filename="output.svg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9999)
