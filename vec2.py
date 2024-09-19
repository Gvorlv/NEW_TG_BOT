#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
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
    #splitter=RecursiveCharacterTextSplitter(chunk_size=750, separators=['#'],)
    

    # fragments = splitter.split_text(text)
    # fragments = splitter.create_documents(fragments)
    # Сохранение фрагментов в текстовый файл в исходном виде
    fragments_path = os.path.join(folder_path, "fragments11.txt")
    with open(fragments_path, 'w', encoding='utf-8') as f:
        for fragment in fragments:
            f.write(str(fragment) + "\n\n")
    return fragments

database=duplicate_headers_without_hashes(document)
#print(database)

source_chunks=split_text(database)
#print(source_chunks)




# Путь к файлам индекса
faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
pkl_file = os.path.join(folder_path, f"{index_name}.pkl")
# api_key = os.environ.get("GPT_SECRET_KEY")
# embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-large")
# db = FAISS.from_documents(source_chunks, embeddings)
# db.save_local(folder_path=folder_path, index_name=index_name)






