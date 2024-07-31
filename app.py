from google.cloud import vision
import io
import os
import pandas as pd

def perform_ocr(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    return document

def extract_text(document):
    text_data = []
    for page in document.pages:
        for block in page.blocks:
            block_text = ""
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    block_text += ''.join([symbol.text for symbol in word.symbols])
                block_text += "\n"
            text_data.append(block_text)
    return "\n".join(text_data)

def ocr_multiple_documents(directory_path):
    all_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            file_path = os.path.join(directory_path, filename)
            document = perform_ocr(file_path)
            text_data = extract_text(document)
            all_texts.append({
                'filename': filename,
                'text': text_data
            })
    return pd.DataFrame(all_texts)

def display_indexed_data(df):
    for index, row in df.iterrows():
        print(f"Document: {row['filename']}")
        print(f"Content: {row['text'][:100]}...")  # Displaying first 100 characters for brevity
        print("\n")

directory_path = '/Users/poornateja/Documents/Documents/OCR/images'
df = ocr_multiple_documents(directory_path)
df.to_csv('ocr_results.csv', index=False)
display_indexed_data(df)
