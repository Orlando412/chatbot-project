from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class chatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: chatRequest):
    client = openai.OpenAI()
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": request.message}]
    )
    return {"response": response.choices[0].message["content"]}
    #return {"response": response["choices"][0]["message"]["content"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)








# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch 

# #initialize fastapi app 
# app = FastAPI()

# #loads pretrained chatbot model
# modelName = "microsoft/DialoGPT-small"
# tokenizer = AutoTokenizer.from_pretrained(modelName)
# model = AutoModelForCausalLM.from_pretrained(modelName)

# def generateResponse(userInput : str) -> str:
#     #generates the chatbot response 
#     input = tokenizer.encode(userInput, + tokenizer.eos_token, return_tensors="pt")
#     output = model.generate(input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(output[:, input.shape[-1]:][0], skip_special_tokens=True)
#     return response

# #request the model
# class chatRequest(BaseModel):
#     message : str
# @app.post("/chat")
# async def chat(request: chatRequest):
#     #chatsbots api endpoint
#     try:
#         response = generateResponse(request.message)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# #run the api 
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0000", port=8000)


