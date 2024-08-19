from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os

import re

import openai
#import tiktoken
#import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Загрузить переменные окружения из файла .env
load_dotenv(dotenv_path="C:/Users/yvorlv/Downloads/NEW_TG_BOT/.env")
folder_path = "C:/Users/yvorlv/Downloads/NEW_TG_BOT/"
index_name = "db_from_texts_PP"
path_to_base="C:/Users/yvorlv/Downloads/NEW_TG_BOT/BazaZnan.txt"
# Загружаем базовый документ
with open(path_to_base, 'r', encoding='utf-8') as file:
    document = file.read()
#print(document)
# API-key
openai.api_key = os.environ.get("GPT_SECRET_KEY")

def duplicate_headers_without_hashes(text):
    """
    Дублирует заголовки в тексте, убирая из дубликатов хэши.

    Например:
    '# Заголовок' превращается в:
    '# Заголовок
    Заголовок'
    """

    # Вспомогательная функция, которая будет вызываться для каждого найденного совпадения в тексте
    def replacer(match):
        # match.group() вернет найденный заголовок с хэшами.
        # затем мы добавляем к нему перенос строки и ту же строку, но без хэшей
        return match.group() + "\n" + match.group().replace("#", "").strip()

    # re.sub ищет в тексте все заголовки, начинающиеся с 1 до 3 хэшей, и заменяет их
    # с помощью функции replacer
    result = re.sub(r'#{1,3} .+', replacer, text)

    return result



def split_text(text):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"),("###", "Header 3"),]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    fragments = markdown_splitter.split_text(text)

    return fragments

database=duplicate_headers_without_hashes(document)
#print(database)

source_chunks=split_text(database)
#print(source_chunks)

# Путь к файлам индекса
faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
pkl_file = os.path.join(folder_path, f"{index_name}.pkl")
api_key = os.environ.get("GPT_SECRET_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(source_chunks, embeddings)
db.save_local(folder_path=folder_path, index_name=index_name)






