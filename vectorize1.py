from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownHeaderTextSplitter
from dotenv import load_dotenv
import os
import re
import openai

# Загрузить переменные окружения из файла .env
load_dotenv(dotenv_path="C:/Users/yvorlv/Downloads/NEW_TG_BOT/.env")

# API-key
openai.api_key = os.environ.get("GPT_SECRET_KEY")

def create_vector_store(path_to_base: str, folder_path: str, index_name: str):
    """
    Создает FAISS векторное хранилище из указанного базового файла.
    """
    # Путь к файлам индекса
    faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_file = os.path.join(folder_path, f"{index_name}.pkl")

    # Получаем API-ключ из окружения
    api_key = os.environ.get("GPT_SECRET_KEY")
    
    if api_key is None:
        raise ValueError("API ключ не найден. Убедитесь, что переменная окружения 'GPT_SECRET_KEY' установлена.")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Проверяем, существуют ли файлы векторной базы данных
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        print("Векторное хранилище уже существует.")
    else:
        # Загружаем базовый документ
        with open(path_to_base, 'r', encoding='utf-8') as file:
            document = file.read()

        # Вспомогательная функция для замены заголовков в тексте
        def replacer(match):
            return match.group() + "\n" + match.group().replace("#", "").strip()
        # Замена заголовков в документе
        result = re.sub(r'#{1,3} .+', replacer, document)

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        fragments = markdown_splitter.split_text(result)

        # Сохранение фрагментов в текстовый файл в исходном виде
        fragments_path = os.path.join(folder_path, "fragments.txt")
        with open(fragments_path, 'w', encoding='utf-8') as f:
            for fragment in fragments:
                f.write(str(fragment) + "\n\n")

        # Создаем и сохраняем векторное хранилище
        #db = FAISS.from_documents(fragments, embeddings)
        #db.save_local(folder_path=folder_path, index_name=index_name)
        
        print("Векторное хранилище создано и сохранено.")
        print(f"Фрагменты сохранены в", fragments_path)

if __name__ == "__main__":
    # Укажите путь к исходному документу
    path_to_base = "C:/Users/yvorlv/Downloads/NEW_TG_BOT/BazaZnan.txt"  # Измените на фактический путь к вашему файлу

    # Укажите папку и имя индекса
    folder_path = "C:/Users/yvorlv/Downloads/NEW_TG_BOT"
    index_name = "db_from_texts_PP"

    # Создаем векторное хранилище
    create_vector_store(path_to_base, folder_path, index_name)
