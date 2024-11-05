from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for JSON data and images
app.mount("/json", StaticFiles(directory="outputs/json"), name="json")
app.mount("/images", StaticFiles(directory="inputs/xkcd_images"), name="images")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse('view_llm_xkcd_explanations.html')

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
