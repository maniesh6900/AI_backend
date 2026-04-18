from fastapi import FastAPI
app = FastAPI()

import os
from dotenv import load_dotenv
load_dotenv()

from LLM_mistralai import generate_text, load_model
from pydantic import BaseModel

MODEL_ID = os.getenv("MODEL")
tokenizer , model = load_model(MODEL_ID)

class Data(BaseModel):
    prompt : str

@app.post("/chat/")
async def get_ans(data : Data):
    r =  generate_text(data.prompt, tokenizer, model)
    print(r)
    return {"msg" : r,  }

@app.post("/check/")
async def getcheck(data : Data):
    return {data : os.getenv("PORT")}


