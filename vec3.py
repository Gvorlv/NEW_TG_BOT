from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
import os
import re
import openai
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Загрузить переменные окружения из файла .env
load_dotenv(dotenv_path="C:/Users/yvorlv/Downloads/NEW_TG_BOT/.env")
folder_path = "C:/Users/yvorlv/Downloads/NEW_TG_BOT/"
index_name = "db_from_texts_PP"
path_to_base = "C:/Users/yvorlv/Downloads/NEW_TG_BOT/BazaZnan.txt"

# Загружаем базовый документ
with open(path_to_base, 'r', encoding='utf-8') as file:
    document = file.read()

# API-key
openai.api_key = os.environ.get("GPT_SECRET_KEY")

def duplicate_headers_without_hashes(text):
    """
    Дублирует заголовки в тексте, убирая из дубликатов хэши.
    """
    def replacer(match):
        return match.group() + "\n" + match.group().replace("#", "").strip()

    result = re.sub(r'#{1,3} .+', replacer, text)
    return result

def split_text_and_create_metadata(text):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"),("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Сегментация текста
    fragments = markdown_splitter.split_text(text)  # Используем split_text вместо create_documents
    
    segmented_chunks = []
    
    for fragment in fragments:
        lower_case_content = fragment.page_content.lower()  # Контент в нижнем регистре
        metadata_content = fragment.page_content  # Оригинальный контент для metadata

        # Создаем чанки с контентом в нижнем регистре и метаданными с оригинальным заголовком
        chunk = {
            'page_content': lower_case_content,
            'metadata': fragment.metadata  # Используем метаданные из объекта Document
        }
        segmented_chunks.append(chunk)
    
    # Сохранение фрагментов в текстовый файл
    fragments_path = os.path.join(folder_path, "fragments11.txt")
    with open(fragments_path, 'w', encoding='utf-8') as f:
        for chunk in segmented_chunks:
            f.write(str(chunk) + "\n\n")
    
    return segmented_chunks

# Подготавливаем текст с заголовками без хэшей
database = duplicate_headers_without_hashes(document)

# Разбиваем текст и создаем чанки с метаданными
source_chunks = split_text_and_create_metadata(database)

# Путь к файлам индекса
faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
pkl_file = os.path.join(folder_path, f"{index_name}.pkl")
api_key = os.environ.get("GPT_SECRET_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-large")

# Создаем базу данных с чанками
db = FAISS.from_documents(
    [Document(page_content=chunk['page_content'], metadata=chunk['metadata']) for chunk in source_chunks], 
    embeddings
)
db.save_local(folder_path=folder_path, index_name=index_name)