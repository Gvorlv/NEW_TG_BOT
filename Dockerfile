# Используем базовый образ Python
FROM python:3.9-slim

# Установка необходимых инструментов и зависимостей
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg

# Установка Poetry
RUN pip install poetry

# Установка рабочей директории
WORKDIR /app

# Копируем файлы Poetry
COPY pyproject.toml poetry.lock /app/

# Установка зависимостей проекта через Poetry
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Копируем все файлы проекта в контейнер
COPY . /app

# Устанавливаем права на директорию с базой данных, если требуется
RUN chmod -R 777 /app/faiss_index

# Открываем необходимые порты (если используются)
EXPOSE 8000

# Команда для запуска приложения
CMD ["poetry", "run", "python", "bot9.py"]
