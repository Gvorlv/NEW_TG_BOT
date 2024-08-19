
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownHeaderTextSplitter
from dotenv import load_dotenv
import os
import re
import openai
folder_path = "C:/Users/yvorlv/Downloads/NEW_TG_BOT/"
index_name = "db_from_texts_PP"
faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
pkl_file = os.path.join(folder_path, f"{index_name}.pkl")

embeddings = OpenAIEmbeddings()

# Проверяем, существуют ли файлы векторной базы данных
if os.path.exists(faiss_file) and os.path.exists(pkl_file):
    # Загружаем векторную базу данных из файлов
    db = FAISS.load_local(folder_path=folder_path, index_name=index_name, embeddings=embeddings)
else:
    # Загружаем базовый документ
    with open(folder_path, 'r', encoding='utf-8') as file:
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
    self.db = FAISS.from_documents(fragments, embeddings)
    
    # Сохраняем векторную базу данных
    self.db.save_local(folder_path=folder_path, index_name=index_name)

# Переменная для хранения саммаризированного диалога
self.dialogue_summary = ""

def insert_newlines(self, text: str, max_len: int = 170) -> str:
"""
Разбивает длинный текст на строки заданной максимальной длины.
"""
words = text.split()
lines = []
current_line = ""
for word in words:
    if len(current_line + " " + word) > max_len:
        lines.append(current_line)
        current_line = ""
    current_line += " " + word
lines.append(current_line)
return "\n".join(lines)

def answer_index(self, system:str, topic:str, temp=0) -> str:
"""
Функция возвращает ответ модели на основе заданной темы.
"""
# находим наиболее релевантные вопросу пользователя чанки:
docs = self.db.similarity_search(topic, k=4)
message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": f"История предыдущего диалога: {self.dialogue_summary} \n  Документ с информацией для ответа пользователю: {message_content}\n\nВопрос пользователя: \n{topic}"}
]

completion = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=messages,
    temperature=temp
)

return completion.choices[0].message.content

def summarize_dialogue(self, dialogue):
"""
Функция возвращает саммаризированный текст диалога.
"""
messages = [
    {"role": "system", "content": "Ты - нейро-саммаризатор. Твоя задача - саммаризировать диалог, который тебе пришел. Если пользователь назвал свое имя, обязательно отрази его в саммаризированном диалоге"},
    {"role": "user", "content": "Саммаризируй следующий диалог консультанта и пользователя: " + dialogue}
]

completion = openai.ChatCompletion.create(
    model="gpt-4o",     # используем gpt4 для более точной саммаризации
    messages=messages,
    temperature=0,          # Используем более низкую температуру для более определенной суммаризации
)
logger.info(f"взврат из summarize_dialogue {completion.choices[0].message.content}")
return completion.choices[0].message.content

def get_answer(self, query: str) -> str:
"""
Функция возвращает ответ на вопрос пользователя и обновляет саммаризацию диалога.
"""
if query=="restart":
    self.dialogue_summary = ""
    return "История диалога удалена"


# Добавляем текущий вопрос к саммаризированному диалогу
self.dialogue_summary += f"\nПользователь: {query}\n"

# Получаем ответ от модели
answer = self.answer_index(default_system, query)

# Добавляем ответ к саммаризированному диалогу
self.dialogue_summary += f"Ассистент: {answer}\n"

# Обновляем саммаризацию диалога
self.dialogue_summary = self.summarize_dialogue(self.dialogue_summary)

return answer