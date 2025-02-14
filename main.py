from fastapi import FastAPI
from pydantic import BaseModel
import openai 
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("API key not found! Set OPENAI_API_KEY as an environment variable.")

class chatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: chatRequest):
    client = openai.OpenAI()
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": request.message}]
    )
    print(f"API Key: {os.getenv('OPENAI_API_KEY')}")
    return {"response": response.choices[0].message["content"]}
    #return {"response": response["choices"][0]["message"]["content"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)












