

<h1 align="center">
  Multi--Agent-PDF-Intelligence-System
</h1>


## 📖 Introduction 

This project develops an advanced Intelligent Document Processing (IDP) system using a multi-agent architecture for PDF document handling and query resolution.

Key Features:
Multi-Agent Collaboration: Specialized AI agents for document processing and query resolution.

PDF Handling: Efficient text extraction and metadata management.

User Interaction: AI-human collaboration for clarity and accuracy.

Applications:

Ideal for legal review, compliance automation, and research data management.

## 🚀 Demo
click on the image to visit the demo video

[![Watch on YouTube](https://img.youtube.com/vi/oMpKkhVB0Eo/hqdefault.jpg)](https://www.youtube.com/watch?v=oMpKkhVB0Eo&t=1s)



## 🛠️ Quick Start 
1. Clone the github repository at the preferable location in your local machine. You will need git to be preinstalled in the system. Once the repository is cloned in your system, with the help of cd command ,
```
git clone https://github.com/piyush1prasad/proj2](https://github.com/raajChit/Multi--Agent-PDF-Intelligence-System.git)
cd Multi--Agent-PDF-Intelligence-System
```

2. This project uses Python 3, so make sure that [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installation/) are preinstalled. All requirements of the project are listed in the ```requirements.txt``` file. Use pip to install all of the requirements.
```
pip install -r requirements.txt
```

3. You can optionally create an environment using venv,
```
python3 -m venv env
source env/bin/activate 
```
4. Use the .env file to store all the api keys.
   
5. You will require google's document AI and translation API using the service account approach.

   Follow the instructions given in the links to get the keys:

   document AI: [https://cloud.google.com/document-ai/docs/setup](url)

   translation API: [https://cloud.google.com/translate/docs/reference/rest](url)

   Make sure to store the json key in the project directory

5. You will also require Groq and OpenAI keys.
   Follow the instructions given in the links to get the keys:
   Groq: [https://console.groq.com/docs/quickstart](url)
   
   OpenAI: [https://platform.openai.com/docs/quickstart](url)
   
6. Once you have set up all the keys. You can run the main file.
```
python3 main.py 
```
7. You can access the API endpoints using Fast API swagger by going to the link : [http://127.0.0.1:8000/docs](url) or [http://localhost:8000/docs](url)









