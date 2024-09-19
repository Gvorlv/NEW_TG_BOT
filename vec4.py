import os
import re
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Загрузить переменные окружения из файла .env
load_dotenv(dotenv_path="C:/Users/yvorlv/Downloads/NEW_TG_BOT/.env")
folder_path = "C:/Users/yvorlv/Downloads/NEW_TG_BOT/"
index_name = "db_from_texts_PP"
path_to_base = "C:/Users/yvorlv/Downloads/NEW_TG_BOT/BazaZnan.txt"

# API-key
openai.api_key = os.environ.get("GPT_SECRET_KEY")

# Загружаем базовый документ
with open(path_to_base, 'r', encoding='utf-8') as file:
    document = file.read()

def duplicate_headers_without_hashes(text):
    """
    Дублирует заголовки в тексте, убирая из дубликатов хэши.
    """
    def replacer(match):
        return match.group() + "\n" + match.group().replace("#", "").strip()

    result = re.sub(r'#{1,3} .+', replacer, text)
    return result

def chunk_text_by_headers(text):
    """
    Разбивает текст на чанки по заголовкам #, ## и ###.
    """
    chunks = []
    current_chunk = ""
    current_metadata = {"Header 1": None, "Header 2": None, "Header 3": None}

    for line in text.splitlines():
        header_match_1 = re.match(r'^# (.*)', line)  # Заголовок 1 уровня
        header_match_2 = re.match(r'^## (.*)', line)  # Заголовок 2 уровня
        header_match_3 = re.match(r'^### (.*)', line)  # Заголовок 3 уровня

        if header_match_1:
            # Сохраняем текущий чанк перед переходом к новому заголовку 1 уровня
            if current_chunk:
                chunks.append((current_chunk.strip(), current_metadata.copy()))
                current_chunk = ""
            # Обновляем заголовок 1 уровня в метаданных
            current_metadata["Header 1"] = header_match_1.group(1)
            current_metadata["Header 2"] = None  # Сбрасываем заголовок 2 уровня
            current_metadata["Header 3"] = None  # Сбрасываем заголовок 3 уровня
        elif header_match_2:
            # Обновляем заголовок 2 уровня в метаданных
            current_metadata["Header 2"] = header_match_2.group(1)
            current_metadata["Header 3"] = None  # Сбрасываем заголовок 3 уровня
        elif header_match_3:
            # Обновляем заголовок 3 уровня в метаданных
            current_metadata["Header 3"] = header_match_3.group(1)
        # Добавляем строку в текущий чанк
        current_chunk += line + "\n"

    # Сохраняем последний чанк, если он есть
    if current_chunk:
        chunks.append((current_chunk.strip(), current_metadata.copy()))

    return chunks

# Подготавливаем текст с заголовками без хэшей
database = duplicate_headers_without_hashes(document)

# Разбиваем текст на чанки по заголовкам
source_chunks = chunk_text_by_headers(database)

# Сохраняем чанки в текстовый файл
def save_chunks_to_file(chunks, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, (chunk, metadata) in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n")
            f.write(chunk + "\n")
            f.write(f"Metadata: {metadata}\n\n")

save_chunks_to_file(source_chunks, os.path.join(folder_path, "fragm1.txt"))

# Путь к файлам индекса
faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
pkl_file = os.path.join(folder_path, f"{index_name}.pkl")
api_key = os.environ.get("GPT_SECRET_KEY")

# Инициализация объекта для работы с embedding'ами
embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-large")

def generate_embeddings_and_save(source_chunks):
    """
    Генерация embedding'ов для документа с чанками и сохранение их в FAISS.
    """
    documents = []
    ids = []

    # Преобразуем каждый чанк в Document и сохраняем его метаданные
    for i, (chunk, metadata) in enumerate(source_chunks):
        lower_case_content = chunk.lower()  # Преобразование текста в нижний регистр

        documents.append(
            Document(
                page_content=lower_case_content,  # Преобразованный текст
                metadata=metadata  # Сохранение метаданных
            )
        )
        ids.append(str(i))  # Присваиваем уникальные идентификаторы чанкам

    # Создание базы данных FAISS с embedding'ами
    db = FAISS.from_documents(documents=documents, embedding=embeddings, ids=ids)

    # Сохранение базы данных FAISS локально
    db.save_local(folder_path=folder_path, index_name=index_name)

# Запуск процесса генерации embedding'ов и сохранения
generate_embeddings_and_save(source_chunks)