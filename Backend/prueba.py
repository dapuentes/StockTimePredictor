from services.model_xgb.main_xgb import *

import uvicorn
from fastapi import FastAPI

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)