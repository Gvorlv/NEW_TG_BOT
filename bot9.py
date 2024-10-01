import os
import re
import sqlite3
from typing import Optional
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydub import AudioSegment
from openai import AsyncOpenAI
from dotenv import load_dotenv
from io import BytesIO
from typing import Dict
from telegram import Update
from telegram.constants import ParseMode
from telegram import Bot
from telegram.constants import ChatAction
import asyncio
import telegram.error
from telegram.ext import (
    Application,
    CommandHandler,
    ChatMemberHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackContext,
    Defaults
)
from telegram import InlineKeyboardMarkup
from pydub.playback import play
import json
from datetime import datetime
from telegram.request import HTTPXRequest  # Импортируем HTTPXRequest для кастомизации

load_dotenv()

# Устанавливаем параметры логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("TOKEN")

# Создаем объект бота без дополнительных параметров таймаута
bot = Bot(token=TOKEN)

clientOAI = AsyncOpenAI()   
MODEL_TURBO_0125 = "gpt-3.5-turbo-0125"
MODEL_VI = "gpt-4-vision-preview"
MODEL_4 = "gpt-4"
model_4o_mini = 'gpt-4o-mini'
model_4o = 'gpt-4o'
model = model_4o_mini

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

url_BZ = ''
temperature = 0
count_type = ''
top_similar_documents = 7
trash_hold = 0 

#embeddings = OpenAIEmbeddings()
#embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system = '''
# Персонаж
Ты-консультант в сети ветеринарных клиник Здоровье животных, которая оказывает ветеринарные услуги и продает ветеринарные препараты и зоотовары. Тебе дано задание отвечать на вопросы операторов коллцентра о регламентах, ветеринарных услугах.
Если пользователь представился и спрашивает как его имя то ответь. 

## Навыки
1. Полный и точный ответ на вопросы пользователя по поводу ветеринарных услуг и регламентов.
2. Основывайся на описаниях регламентов и документах.
3. Используй ссылки, если они указаны в документах.
4. Обращайся к пользователю по имени, если оно известно.
5. Способность полностью излагать текст и все пункты и подпункты пронумерованных пунктов из регламентов.
6. Здороваться с пользователем, если он поздоровался.
7. Не здороваться в ответ, если пользователь не поздоровался.
8. Имена, фамилии и отчества, а так же названия улиц в ответе пиши с большой буквы.

## Ограничения
- Отвечай только на вопросы, касающиеся регламентов.
- Используй язык оригинального запроса от пользователя.
- Если ссылок нет, ничего от себя не придумывай.
'''

system_analyser = ''' 
Ты - нейро-саммаризатор. Твоя задача - саммаризировать диалог, который тебе пришел. 
Если пользователь назвал свое имя, обязательно отрази его в саммаризированном диалоге.
'''

Contactnie_dannie = """
*Контактные данные*

1. *Воронеж*
   📍 Ветклиника, ветаптека: ул. Волгоградская 44
   🕒 Пн-Вс 24 часа
   ☎️ +7(473)300-34-50
   📍 Ветклиника, ветаптека: ул. Кольцовская, 49
   🕒 Пн-Вс 24 часа
   ☎️ +7(473)300-34-50
   📧 @zdorovet.ru
"""

def escape_markdown(text: str):
    """
    Escape special characters for Telegram MarkdownV2.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# Создаем базу данных для хранения вопросов и ответов
def create_database():
    conn = sqlite3.connect('questions.db')
    c = conn.cursor()
    # Создаем таблицу для хранения вопросов, ответов и оценок
    c.execute('''CREATE TABLE IF NOT EXISTS questions (
                 id INTEGER PRIMARY KEY,
                 user_id INTEGER,
                 username TEXT,   
                 question TEXT,
                 question_time DATETIME,
                 answer TEXT,
                 answer_time DATETIME,
                 rating INTEGER)''')
    conn.commit()
    conn.close()

create_database()

# Функция для сохранения вопроса пользователя в базу данных
def save_question(user_id: int, question: str):
    conn = sqlite3.connect('questions.db')
    c = conn.cursor()
    # Вставляем новый вопрос в таблицу
    c.execute('''INSERT INTO questions (user_id, question, question_time)
                 VALUES (?, ?, datetime('now'))''', (user_id, question))
    conn.commit()
    question_id = c.lastrowid
    conn.close()
    return question_id

# Функция для сохранения ответа на вопрос
def save_answer(question_id: int, answer: str):
    conn = sqlite3.connect('questions.db')
    c = conn.cursor()
    # Обновляем ответ на вопрос
    c.execute('''UPDATE questions
                 SET answer = ?, answer_time = datetime('now')
                 WHERE id = ?''', (answer, question_id))
    conn.commit()
    conn.close()

# Функция для сохранения оценки ответа
def save_rating(question_id: int, rating: int):
    conn = sqlite3.connect('questions.db')
    c = conn.cursor()
    # Обновляем оценку для вопроса
    c.execute('''UPDATE questions
                 SET rating = ?
                 WHERE id = ?''', (rating, question_id))
    conn.commit()
    conn.close()

# Путь к базе данных FAISS и имя индекса
db_path = 'faiss_index'
index_name = "db_from_texts_PP"
new_db = FAISS.load_local(db_path, embeddings, index_name, allow_dangerous_deserialization=True)

# Асинхронная функция для получения ответа от модели
async def get_answer(text: str, gpt_context: str='', clients_goods: str='')->str: ####
    logger.info(f'запуск функции [get_answer] {text}')
    # Поиск схожих документов по контексту
    docs = []
    docs_direct = await new_db.asimilarity_search_with_score(text, include_metadata=True, k=top_similar_documents)
    docs_direct = [doc for doc, score in docs_direct if score > trash_hold]
    docs.extend(docs_direct)
    # Формируем контент вопроса и запрашиваем у модели ответ
    message_content = ' '.join([f'\nОписание регламентов №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)])
    question_content = f"Ответь на вопрос пользователя на основании представленных описаний регламентов. Документы с описаниями регламентов для ответа клиенту: {message_content}\n\n"
    question_content_2 = question_content + f"Текущий вопрос пользователя: \n{text}"
    text_user = question_content_2
    print(f'{text_user} \n')
    # Проверка и добавление контекста
    if len(gpt_context) > 0:
        text_user = question_content + "\n\n" + gpt_context + "\n\n" + f"Актуальный вопрос пользователя: \n{text}."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text_user}
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text_user}
        ]
    
    print("Сообщения для API OpenAI:", messages)
    
    try:
        # Запрос к модели OpenAI
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        print("Ответ API OpenAI:", completion)
    except Exception as e:
        print(f'Произошла ошибка: {e}')
        return "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте снова позже."
    
    answer_text = completion.choices[0].message.content
    print(f'{answer_text} \n')
    return answer_text
# Определение роли системы и навыков для саммаризации диалогов
system_summarize = '''# Character
Ты - нейро-саммаризатор. Твоя задача - саммаризировать диалог, который тебе пришел. 

## Skills
### Skill 1: 
- Саммаризировать диалог, сохраняя ключевые моменты и имена, если упомянуты.

### Skill 2:
- Понимать запросы пользователя о более подробной информации по определенным регламентам.

## Constraints
- Соблюдать формат кратких саммаризаций.
- Отражать имена пользователей, если упомянуты.'''

# Асинхронная функция для создания саммари на основе диалога
async def summarize_questions(dialog: str, context: str)->str:
    messages = [
        {"role": "system", "content": system_summarize},
        {"role": "user", "content": "Саммаризируй следующий диалог консультанта и пользователя: " + " ".join(dialog)}
    ]

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    #print("*****1111****,", type(completion.choices[0].message.content))
    return completion.choices[0].message.content

# Асинхронная функция для создания ответа 
async def create_completion(model, system, content, temperature: int=0):
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': content}
    ]
    completion = await client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature
    )
    return completion.choices[0].message.content

# Асинхронная функция для анализа интересующих вопросов
async def run_model_analyser(topic:str, existing_goods: Optional[str]) -> str:
    logger.info(f'Параметр existing_goods функции run_model_analyser {existing_goods}')
    #content = f'Выдели в сообщении клиента интересующие его вопросы по регламентам. Выдели только те вопросы, которые не входят в список "{existing_goods}". Перечисли вопросы через запятую. Если новых в списке нет, выведи пустое значение. То, что говорил клиент: ' + " ".join(topic)
    content = f'Выдели в сообщении клиента интересующие его вопросы по регламентам.  Перечисли вопросы через запятую.  То, что говорил клиент: ' + " ".join(topic)
    return await create_completion(model, system_analyser, content, 0)

# Функция для повторной отправки сообщения при возникновении таймаута
async def retry_send_message(context, chat_id, text, reply_to_message_id: Optional[str] = None, parse_mode=None, max_retries=3):
    """
    Функция для повторной отправки сообщения при возникновении таймаута.
    """
    for attempt in range(max_retries):
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id, parse_mode=parse_mode)
            break  # Выход из цикла, если отправка успешна
        except telegram.error.TimedOut as e:
            logger.error(f"Попытка {attempt + 1}: ошибка таймаута при отправке сообщения: {e}")
            await asyncio.sleep(2)  # Задержка перед повторной попыткой
        except Exception as e:
            logger.error(f"Не удалось отправить сообщение: {e}")
            break

from telegram.ext import filters

# Обработка текстового сообщения пользователя
async def gpt(update: Update, context: CallbackContext):
    logger.info(f'запуск функции [gpt] это update-{update}, это context-{context}')
    clients_goods = ""
    user_id = update.message.from_user.id
    question = update.message.text

    # Проверка на наличие оценки в контексте
    if question.isdigit() and 'current_question_id' in context.user_data:
        await handle_rating(update, context)
        return
    
    question_id = save_question(user_id, question)
   
    if context.bot_data[update.message.from_user.id]['voprosi'] > 0:
        first_message = await update.message.reply_text(
            'Ваш запрос обрабатывается, пожалуйста подождите...',
            reply_to_message_id=update.message.message_id,
            reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
        )

        goods = await run_model_analyser(question, clients_goods)
        if goods:
            godds_arr = goods.replace(",", " ").replace("  ", " ").split(' ')
            for good in godds_arr:
                if good and good not in clients_goods:
                    clients_goods += ' ' + good

        if context.bot_data[update.message.from_user.id]['text'] != '':
            summarized_history = "Вот краткий обзор предыдущего диалога: " + await summarize_questions({context.bot_data[update.message.from_user.id]['text'].replace('<marker>','')}, context)
            res = await get_answer(question, summarized_history, clients_goods)
        else:
            res = await get_answer(question, '', clients_goods)
        
        context.bot_data[update.message.from_user.id]['text'] += f"Клиент: {question}\nChatGPT: {res}\n\n<marker>"
        res += "\n" + Contactnie_dannie
        res = escape_markdown(res)

        await context.bot.edit_message_text(
            text=res,
            chat_id=update.message.chat_id,
            message_id=first_message.message_id,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
        )

        # Сохранение ответа в базу данных
        save_answer(question_id, res)

        # Запрос оценки у пользователя
        await context.bot.send_message(
            chat_id=update.message.chat_id,
            text="Пожалуйста, оцените качество ответа от 1 до 3 (1-информация не найдена, 2-информация присутствует но не верна, 3-информация верна).",
            reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
        )
        
        # Сохранение идентификатора вопроса в контексте для дальнейшей оценки
        context.user_data['current_question_id'] = question_id

        context.bot_data[update.message.from_user.id]['voprosi'] -= 1
    else:
        await update.message.reply_text(
            'Ваши запросы на сегодня исчерпаны',
            reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
        )

# # Обработка и сохранение оценки от пользователя
# async def handle_rating(update: Update, context: CallbackContext):
#     rating_text = update.message.text
#     question_id = context.user_data.get('current_question_id')
    
#     # Проверка на наличие оценки в контексте
#     if question_id is not None:
#         try:
#             rating = int(rating_text)
#             if rating in [1, 2, 3]:
#                 save_rating(question_id, rating)
#                 await update.message.reply_text(
#                     "Спасибо за вашу оценку!",
#                     reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
#                 )
#             else:
#                 await update.message.reply_text(
#                     "Пожалуйста, введите оценку в формате (от 1 до 3).",
#                     reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
#                 )
#         except ValueError:
#             await update.message.reply_text(
#                 "Пожалуйста, введите числовое значение от 1 до 3.",
#                 reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
#             )
#     else:
#         await update.message.reply_text(
#             "Не найден вопрос для оценки. Пожалуйста, задайте новый вопрос.",
#             reply_markup=InlineKeyboardMarkup([])  # Удаление старых кнопок
#         )
#     # Очистка текущего вопроса из контекста
#     context.user_data['current_question_id'] = None

#обработка голосового сообщения
async def gpt_v(update, context):
    clients_goods = ""
    user_id = update.message.from_user.id
    if context.bot_data[update.message.from_user.id]['voprosi'] > 0:
        first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...', reply_to_message_id=update.message.message_id)
    
        file = await update.message.voice.get_file() # Получаем файл с помощью метода
        voice_as_byte = await file.download_as_bytearray() # Конвертируем file в массив байт

        byte_voice = BytesIO(voice_as_byte) #Оборачиваем voice_as_byte в объект BytesIO
        audio = AudioSegment.from_file(byte_voice, format='ogg') # Преобразуем в аудиофайл с помощью библиотеки pydub 
        audio.export('voice_message.mp3', format='mp3') # Сохраняем audio в аудиофайл voice_message.mp3
        audio_file = open("voice_message.mp3", "rb")
        text = await clientOAI.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        ) # MP3-файл передаётся модели OpenAI Whisper для транскрипции речи в текст
    
        question_id = save_question(user_id, text)

        goods = await run_model_analyser(text, clients_goods)
        if goods:
            godds_arr = goods.replace(",", " ").replace("  ", " ").split(' ')
            for good in godds_arr:
                if good and not good in clients_goods:
                    clients_goods += ' ' + good 
    
        if context.bot_data[update.message.from_user.id]['text'] != '':
            summarized_history = "Вот краткий обзор предыдущего диалога: " + await summarize_questions({context.bot_data[update.message.from_user.id]['text'].replace('<marker>','')}, context) 
            res = await get_answer(text, summarized_history, clients_goods)
        else:
            res = await get_answer(text, '', clients_goods)
        
        context.bot_data[update.message.from_user.id]['text'] += f"Клиент: {text}\nChatGPT: {res}\n\n<marker>"

        res = res + "\n" + Contactnie_dannie
        res = escape_markdown(res)

        await context.bot.edit_message_text(text=res, chat_id=update.message.chat_id, message_id=first_message.message_id)

        # Сохранение ответа в базу данных
        save_answer(question_id, res)

        # Запрос оценки у пользователя
        await context.bot.send_message(chat_id=update.message.chat_id, text="Пожалуйста, оцените качество ответа от 1 до 3 (1-информация не найдена, 2-информация присутствует но не верна, 3-информация верна).")
        
        # Сохранение идентификатора вопроса в контексте для дальнейшей оценки
        context.user_data['current_question_id'] = question_id

        context.bot_data[update.message.from_user.id]['voprosi'] -= 1
    else:
        await update.message.reply_text('Ваши запросы на сегодня исчерпаны')

# Обработка и сохранение оценки от пользователя
async def handle_rating(update: Update, context: CallbackContext):
    rating_text = update.message.text
    question_id = context.user_data.get('current_question_id')
    
    if question_id is not None:
        try:
            rating = int(rating_text)
            if rating in [1, 2, 3]:
                save_rating(question_id, rating)
                await update.message.reply_text("Спасибо за вашу оценку!")
            else:
                await update.message.reply_text("Пожалуйста, введите оценку в правильном формате (от 1 до 3).")
        except ValueError:
            await update.message.reply_text("Пожалуйста, введите числовое значение от 1 до 3.")
    else:
        await update.message.reply_text("Не найден вопрос для оценки. Пожалуйста, задайте новый вопрос.")


# Функция запуска и авторизации пользователя
async def start(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    username = update.message.from_user.username or update.message.from_user.first_name
    # Load authorized user IDs from access.txt
    with open('access.txt', 'r') as file:
        authorized_users = file.read().splitlines()

    if str(user_id) not in  authorized_users:
    # Log unauthorized access attempt
        with open('users.txt', 'a') as file:
            file.write(f"{datetime.now()} - Принята команда /start от user_id: {user_id} (username: {username})\n")
        
        await update.message.reply_text('Обратитесь к администратору для авторизации')
        return

    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = {}
        context.bot_data[update.message.from_user.id]['voprosi'] = 35
        context.bot_data[update.message.from_user.id]['text'] = ''

    user = update.effective_user
    await update.message.reply_html(f"Добро пожаловать, {user.mention_html()}! Это консультант по регламентам Здоровье животных. Ты можешь мне задать любой вопрос по данной теме.")

# Сброс данных пользователя
async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context.bot_data[update.message.from_user.id]
    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = {}
        context.bot_data[update.message.from_user.id]['voprosi'] = 35
        context.bot_data[update.message.from_user.id]['text'] = ''
    await update.message.reply_text('Данные очищены')

# Сохранение данных в файл
async def data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with open ('data.json', 'w', encoding='utf-8') as fp:
        json.dump(context.bot_data, fp)
    await update.message.reply_text('Данные загружены')  

# Вспомогательная функция помощи
async def help(update, context):
    user = update.effective_user
    await update.message.reply_html(f'Hi, {user.mention_html()}! Поговори с консультантом, если тебе нужна помощь по регламентам')

# Приветственное сообщение для нового пользователя
async def welcome_message(update, context):
    new_user = update.message.new_chat_members[0]
    await update.message.reply_text(chat_id=update.effective_chat.id,
        text=f'Добро пожаловать, {new_user.first_name}! Я консультант по регламентам. Нажмите на /start для начала использования бота.')
    context.bot.pin_chat_message(chat_id=update.effective_chat.id,
        message_id=update.message.message_id)

# Отображение статуса запросов пользователя
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Осталось запросов: {context.bot_data[update.message.from_user.id]["voprosi"]}')

# Ежедневное обновление запросов пользователей
async def callback_daily(context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data != {}:
            for key in context.bot_data:
                context.bot_data[key]['voprosi'] = 20
            print('Запросы пользователей обновлены')
    else:
            print('Не найдено ни одного пользователя')

# Главная функция запуска бота
def main():
    request = HTTPXRequest(http_version="1.1", connect_timeout=30.0, read_timeout=30.0)
    application = Application.builder().token(TOKEN).request(request).build()

    job_queue = application.job_queue
    job_queue.run_repeating(callback_daily, interval=43200, first=60)
    
    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(CommandHandler("help", help, block=False))
    application.add_handler(CommandHandler("status", status, block=False))
    application.add_handler(CommandHandler("data", data, block=False))
    application.add_handler(CommandHandler("reset", reset, block=False))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.REPLY, gpt, block=False))
    application.add_handler(MessageHandler(filters.VOICE, gpt_v, block=False))
    application.add_handler(MessageHandler(filters.Regex(r'^\d+$'), handle_rating, block=False))
    application.add_handler(ChatMemberHandler(welcome_message, ChatMemberHandler.CHAT_MEMBER))
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    print('Бот остановлен')

if __name__ == '__main__':
    main()