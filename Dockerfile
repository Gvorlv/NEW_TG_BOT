# Dockerfile
FROM python:3.9-slim

# Установка Poetry
RUN pip install poetry

# Установка зависимостей проекта
WORKDIR /NEW_TG_BOT
COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Копирование исходного кода
COPY . /app

# Команда для запуска приложения
CMD ["poetry", "run", "python", "bot9.py"]  # Замените "main.py" на основной файл вашего проекта