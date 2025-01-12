import config
import os
from pypdf import PdfReader, PdfWriter

def parse_and_translate_pdf(target_language, read_file_location, write_file_name):
    # Parse the PDF and translate the extracted text
    parsed_text = process_pdf_with_processor(
        config.PROJECT_ID,
        config.LOCATION,
        config.PROCESSOR_ID,
        read_file_location
    )
    translate_text(target_language, parsed_text,write_file_name)


def process_pdf_with_processor(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str
):
    
    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai
    """
    Uses an existing processor to parse a PDF file via Document AI.
    """
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # Build the processor resource name
    processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    # Read the PDF into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Prepare the raw document
    raw_document = documentai.RawDocument(
        content=image_content, mime_type="application/pdf"
    )

    # Construct the request
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
    )

    print("\n\nPARSING DOCUMENT")
    # Send the request
    result = client.process_document(request=request)
    document_object = result.document

    print("\n\nDOCUMENT TEXT PARSED")
    print("\n\nSENDING FOR TRANSLATION")
    return document_object.text




def translate_text(target_language: str, text: str,  filename: str, folder: str = "processed_files"):
    
    from google.cloud import translate_v2

    translate_client = translate_v2.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    detection = translate_client.detect_language(text)
    source_lang = detection["language"]

    if source_lang != target_language:
        translation = translate_client.translate(
            text, target_language=target_language
        )
        result =  translation["translatedText"]

    else:
        result = text
        

    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(result)
   

    print(f"\n\nTranslation saved to {file_path}")
    print(f"\nDetected source language: {source_lang}")

def split_pdf(file_path, split_folder, max_pages=15):

    reader = PdfReader(file_path)
    total_pages = len(reader.pages)

    for start in range(0, total_pages, max_pages):
        writer = PdfWriter()
        for i in range(start, min(start + max_pages, total_pages)):
            writer.add_page(reader.pages[i])

        # Save the split file in the split_files folder
        split_file_path = os.path.join(split_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_part_{start//max_pages + 1}.pdf")
        with open(split_file_path, "wb") as output_file:
            writer.write(output_file)

    print(f"PDF split into {len(os.listdir(split_folder))} parts in '{split_folder}'.")
