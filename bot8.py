import os
import re
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydub import AudioSegment
from openai import AsyncOpenAI
from dotenv import load_dotenv
from io import BytesIO
from gtts import gTTS
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
from pydub.playback import play
import requests
import pickle
import pprint
import json
import aiohttp
import httpx
import time
from datetime import datetime
from telegram.request import HTTPXRequest  # Импортируем HTTPXRequest для кастомизации

load_dotenv()

##########
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("TOKEN")
GPT_SECRET_KEY = os.environ.get("GPT_SECRET_KEY")
os.environ["OPENAI_API_KEY"] = GPT_SECRET_KEY

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
top_similar_documents = 10
trash_hold = 0 

embeddings = OpenAIEmbeddings()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system = '''
# Персонаж
Ты-консультант в сети ветеринарных клиник Здоровье животных, которая оказывает ветеринарные услуги и продает ветеринарные препараты и зоотовары. Тебе дано задание отвечать на вопросы операторов коллцентра о регламентах, ветеринарных услугах. 

## Навыки
1. Полный и точный ответ на вопросы пользователя по поводу ветеринарных услуг и регламентов.
2. Основывайся на описаниях регламентов и документах.
3. Используй ссылки, если они указаны в документах.
4. Обращайся к пользователю по имени, если оно известно.
5. Способность полностью излагать текст и все пункты и подпункты пронумерованных пунктов из регламентов.
6. Здороваться с пользователем, если он поздоровался.
7. Не здороваться в ответ, если пользователь не поздоровался.

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

def escape_markdown(text):
    """
    Escape special characters for Telegram MarkdownV2.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

async def get_answer(text, gpt_context='', clients_goods=''):
    db_path = 'faiss_index'
    index_name = "db_from_texts_PP"
    new_db = FAISS.load_local(db_path, embeddings, index_name, allow_dangerous_deserialization=True)
    
    if clients_goods:
        docs = await new_db.asimilarity_search_with_score(clients_goods, k=top_similar_documents)
    else:
        docs = []

    docs = [doc for doc, score in docs if score > trash_hold]
    docs_direct = await new_db.asimilarity_search_with_score(text, k=top_similar_documents)
    docs_direct = [doc for doc, score in docs_direct if score > trash_hold]
    docs.extend(docs_direct)
    
    message_content = ' '.join([f'\nОписание регламентов №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)])
    question_content = f"Ответь на вопрос пользователя на основании представленных описаний регламентов. Документы с описаниями регламентов для ответа клиенту: {message_content}\n\n"
 
    question_content_2 = question_content + f"Текущий вопрос пользователя: \n{text}"
    
    text_user = question_content_2
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
    return answer_text

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

async def summarize_questions(dialog, context):
    messages = [
        {"role": "system", "content": system_summarize},
        {"role": "user", "content": "Саммаризируй следующий диалог консультанта и пользователя: " + " ".join(dialog)}
    ]

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )

    return completion.choices[0].message.content

async def create_completion(model, system, content, temperature):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content}
    ]
    
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return completion.choices[0].message.content

async def run_model_analyser(topic, existing_goods):
    content = f"Выдели в сообщении клиента интересующие его вопросы по регламентам. Выдели только те товары, которые не входят в список \"{existing_goods}\". Перечисли товары через запятую. Если новых в списке нет, выведи пустое значение. То, что говорил клиент: " + " ".join(topic)
    return await create_completion(model, system_analyser, content, 0)

async def retry_send_message(context, chat_id, text, reply_to_message_id=None, parse_mode=None, max_retries=3):
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

async def gpt(update: Update, context: CallbackContext):
    clients_goods = ""
   
    if context.bot_data[update.message.from_user.id]['voprosi'] > 0:
        try:
            first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...', reply_to_message_id=update.message.message_id)
        except telegram.error.TimedOut as e:
            logger.error(f"Ошибка таймаута при отправке сообщения: {e}")
            await retry_send_message(context, update.message.chat_id, 'Произошла ошибка при отправке сообщения. Пожалуйста, попробуйте снова.', reply_to_message_id=update.message.message_id)
            return
        
        print("Получено сообщение от пользователя:", update.message.text)
        
        goods = await run_model_analyser(update.message.text, clients_goods)
        print("Идентифицированные вопросы:", goods)
        
        if goods:
            godds_arr = goods.replace(",", " ").replace("  ", " ").split(' ')
            for good in godds_arr:
                if good and good not in clients_goods:
                    clients_goods += ' ' + good
        
        if context.bot_data[update.message.from_user.id]['text'] != '':
            summarized_history = "Вот краткий обзор предыдущего диалога: " + await summarize_questions({context.bot_data[update.message.from_user.id]['text'].replace('<marker>','')}, context)
            print("Краткий обзор истории:", summarized_history)
            
            res = await get_answer(update.message.text, summarized_history, clients_goods)
        else:
            full_answer = update.message.text
            res = await get_answer(full_answer, '', clients_goods)
        
        print("Ответ от get_answer:", res)
        
        if len(context.bot_data[update.message.from_user.id]['text'].split('<marker>')) < 5:
            context.bot_data[update.message.from_user.id]['text'] += f"Клиент: {update.message.text}\n"
            context.bot_data[update.message.from_user.id]['text'] += f"ChatGPT: {res}\n\n<marker>"
        else:
            context.bot_data[update.message.from_user.id]['text'] = '<marker>'.join(context.bot_data[update.message.from_user.id]['text'].split('<marker>')[1:])
            context.bot_data[update.message.from_user.id]['text'] += f"Клиент: {update.message.text}\n"
            context.bot_data[update.message.from_user.id]['text'] += f"ChatGPT: {res}\n\n<marker>"

        res = res + "\n" + Contactnie_dannie
        res = escape_markdown(res)  # Экранируем специальные символы

        try:
            await context.bot.edit_message_text(text=res, chat_id=update.message.chat_id, message_id=first_message.message_id, parse_mode=ParseMode.MARKDOWN_V2)
        except telegram.error.TimedOut as e:
            logger.error(f"Ошибка таймаута при редактировании сообщения: {e}")
            await retry_send_message(context, update.message.chat_id, 'Произошла ошибка при редактировании сообщения. Пожалуйста, попробуйте снова.', reply_to_message_id=update.message.message_id)
            return
        
        context.bot_data[update.message.from_user.id]['voprosi'] -= 1
    else:
        await update.message.reply_text('Ваши запросы на сегодня исчерпаны')

async def gpt_v(update, context):
    clients_goods = ""
    
    if context.bot_data[update.message.from_user.id]['voprosi'] > 0:
        try:
            first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...', reply_to_message_id=update.message.message_id)
        except telegram.error.TimedOut as e:
            logger.error(f"Ошибка таймаута при отправке сообщения: {e}")
            await retry_send_message(context, update.message.chat_id, 'Произошла ошибка при отправке сообщения. Пожалуйста, попробуйте снова.', reply_to_message_id=update.message.message_id)
            return
        
        file = await update.message.voice.get_file()
        voice_as_byte = await file.download_as_bytearray()

        byte_voice = BytesIO(voice_as_byte)
        audio = AudioSegment.from_file(byte_voice, format='ogg')
        audio.export('voice_message.mp3', format='mp3') 
        audio_file = open("voice_message.mp3", "rb")
        text = await clientOAI.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
        
        goods = await run_model_analyser(text, clients_goods)
        if goods:
            godds_arr = goods.replace(",", " ").replace("  ", " ").split(' ')
            for good in godds_arr:
                if good and not good in clients_goods:
                    clients_goods += ' ' + good 
        
        if context.bot_data[update.message.from_user.id]['text'] != '':
            summarized_history = "Вот краткий обзор предыдущего диалога: " + await summarize_questions({context.bot_data[update.message.from_user.id]['text'].replace('<marker>','')}, context) 
            full_answer = summarized_history + "\n\nТекущий вопрос: " + text
            res = await get_answer(text, summarized_history, clients_goods)
        else:
            full_answer = text
            res = await get_answer(full_answer, '', clients_goods)
            
        if len(context.bot_data[update.message.from_user.id]['text'].split('<marker>')) < 5:
            context.bot_data[update.message.from_user.id]['text'] += f"Клиент: {text}\n"
            context.bot_data[update.message.from_user.id]['text'] += f"ChatGPT: {res}\n\n<marker>"
        else:
            context.bot_data[update.message.from_user.id]['text'] = '<marker>'.join(context.bot_data[update.message.from_user.id]['text'].split('<marker>')[1:])
            context.bot_data[update.message.from_user.id]['text'] += f"Клиент: {text}\n"
            context.bot_data[update.message.from_user.id]['text'] += f"ChatGPT: {res}\n\n<marker>"

        res = res + "\n" + Contactnie_dannie
        res = escape_markdown(res)  # Экранируем специальные символы

        try:
            await context.bot.edit_message_text(text=res, chat_id=update.message.chat_id, message_id=first_message.message_id)
        except telegram.error.TimedOut as e:
            logger.error(f"Ошибка таймаута при редактировании сообщения: {e}")
            await retry_send_message(context, update.message.chat_id, 'Произошла ошибка при редактировании сообщения. Пожалуйста, попробуйте снова.', reply_to_message_id=update.message.message_id)
            return

        context.bot_data[update.message.from_user.id]['voprosi'] -= 1
    else:
        await update.message.reply_text('Ваши запросы на сегодня исчерпаны')

async def start(update, context):
    user_id = update.message.from_user.id
    username = update.message.from_user.username or update.message.from_user.first_name
    # Load authorized user IDs from access.txt
    with open('access.txt', 'r') as file:
        authorized_users = file.read().splitlines()

    if str(user_id) not in authorized_users:
        # Log unauthorized access attempt
        with open('users.txt', 'a') as file:
            file.write(f"{datetime.now()} - Received /start command from user_id: {user_id} (username: {username})\n")
        
        await update.message.reply_text('Обратитесь к администратору для авторизации')
        return

    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = {}
        context.bot_data[update.message.from_user.id]['voprosi'] = 35
        context.bot_data[update.message.from_user.id]['text'] = ''
    
    user = update.effective_user
    await update.message.reply_html(f"Добро пожаловать, {user.mention_html()}! Это консультант по регламентам Здоровье животных. Ты можешь мне задать любой вопрос по данной теме.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context.bot_data[update.message.from_user.id]
    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = {}
        context.bot_data[update.message.from_user.id]['voprosi'] = 35
        context.bot_data[update.message.from_user.id]['text'] = ''
    
    await update.message.reply_text('Данные очищены')

async def data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with open('data.json', 'w', encoding='utf-8') as fp:
        json.dump(context.bot_data, fp)
    
    await update.message.reply_text('Данные загружены')  

async def help(update, context):
    user = update.effective_user
    await update.message.reply_html(f"Hi, {user.mention_html()}! Поговори с консультантом, если тебе нужна помощь по строительному оборудованию и материалам")

async def welcome_message(update, context):
    new_user = update.message.new_chat_members[0]
    await update.message.reply_text(chat_id=update.effective_chat.id,
                             text=f"Добро пожаловать, {new_user.first_name}! Я консультант по строительному оборудованию и материалам. Нажмите на /start для начала использования бота.")
    context.bot.pin_chat_message(chat_id=update.effective_chat.id,
                                  message_id=update.message.message_id)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Осталось запросов: {context.bot_data[update.message.from_user.id]['voprosi']}")

async def callback_daily(context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data != {}:
        for key in context.bot_data:
            context.bot_data[key]['voprosi'] = 20
        print('Запросы пользователей обновлены')
    else:
        print('Не найдено ни одного пользователя')

def main():
    # Устанавливаем кастомный HTTPXRequest с таймаутом
    request = HTTPXRequest(http_version="1.1", connect_timeout=30.0, read_timeout=30.0)  # Увеличили таймауты
    
    application = Application.builder().token(TOKEN).request(request).build()
    
    print('Бот запущен...')

    job_queue = application.job_queue
    job_queue.run_repeating(callback_daily,
                            interval=43200,
                            first=60)
    
    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(CommandHandler("help", help, block=False))
    application.add_handler(CommandHandler("status", status, block=False))
    application.add_handler(CommandHandler("data", data, block=False))
    application.add_handler(CommandHandler("reset", reset, block=False))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.REPLY, gpt, block=False))
    application.add_handler(MessageHandler(filters.VOICE, gpt_v, block=False))
    application.add_handler(ChatMemberHandler(welcome_message, ChatMemberHandler.CHAT_MEMBER))
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    print('Бот остановлен')

if __name__ == "__main__":
    main()
