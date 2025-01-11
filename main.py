import os
from typing import List
from fastapi import FastAPI, File, UploadFile
from parsing import parse_and_translate_pdf
from indexing import index_files
from agent_process import agent_process
import config
import uvicorn
import shutil



app = FastAPI()


@app.post("/uploadpdf")
async def upload_pdf(files: List[UploadFile] = File(...), target_language: str = "en"):
    os.makedirs("temp_files", exist_ok=True)

    try:
        for file in files:
                file_location = f"./temp_files/{file.filename}"
                file_name_without_ext = os.path.splitext(file.filename)[0]
                write_file_name = f"{file_name_without_ext}.txt"
                with open(file_location, "wb") as f:
                    content = await file.read()
                    f.write(content)

                parse_and_translate_pdf(target_language=target_language,read_file_location=file_location, write_file_name=write_file_name)
                index_files()
                
        shutil.rmtree("temp_files")
        shutil.rmtree("processed_files")
        os.makedirs("temp_files", exist_ok=True)
        os.makedirs("processed_files", exist_ok=True)

        return {"result":"success", "remarks":"PDF files parsed and indexed"}
            
    except Exception as e:
        raise e
    
chat_history=[]
@app.post("/ask")
async def ask_question(request: str):
    try:
        result = agent_process(request, chat_history)
        return result
    except Exception as e:
        raise e
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


























@app.get("/")
def read_root():
    """
    Checks if all environment variables are set.
    If any are missing, returns an error message with those variables.
    Otherwise, returns each variable's value.
    """
    missing_vars = []

    if not config.PROJECT_ID:
        missing_vars.append("PROJECT_ID")
    if not config.LOCATION:
        missing_vars.append("LOCATION")
    if not config.PROCESSOR_ID:
        missing_vars.append("PROCESSOR_ID")
    if not config.CREDENTIALS_PATH:
        missing_vars.append("GOOGLE_APPLICATION_CREDENTIALS")

    if missing_vars:
        return {
            "error": "The following environment variables are not set:",
            "missing_variables": missing_vars
        }

    return {
        "project_id": config.PROJECT_ID,
        "location": config.LOCATION,
        "processor_id": config.PROCESSOR_ID,
        "credentials_file": config.CREDENTIALS_PATH
    }


