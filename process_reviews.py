import sqlite3
import pandas as pd
import re
import csv
import io
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

# --- НАСТРОЙКА NLP ---
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

STOP_WORDS = {"и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "аптека", "это", "который"}

def get_season(date_str):
    date_str = date_str.lower()
    if any(m in date_str for m in ['декабр', 'январ', 'феврал']): return 'Зима'
    if any(m in date_str for m in ['март', 'апрел', 'май']): return 'Весна'
    if any(m in date_str for m in ['июн', 'июл', 'август']): return 'Лето'
    if any(m in date_str for m in ['сентябр', 'октябр', 'ноябр']): return 'Осень'
    return 'Не определен'

def process_review(row_string):
    """
    Парсит строку CSV, лемматизирует текст и возвращает список слов с метаданными.
    Добавлена обработка ошибок и отсутствующих полей.
    """
    f = io.StringIO(row_string)
    reader = csv.reader(f, delimiter=',')
    try:
        parts = next(reader)
        
        # Проверяем, достаточно ли полей в строке
        if len(parts) < 5:
            # print(f"Пропускаю строку (недостаточно полей): {row_string.strip()}")
            return []
        
        # Определяем данные, используя безопасный доступ по индексу
        date_text = parts[2].strip() if len(parts) > 2 else ""
        rating_str = parts[3].strip() if len(parts) > 3 else ""
        review_text = parts[4].strip() if len(parts) > 4 else ""

        # Если текста отзыва нет, пропускаем строку (это, видимо, дубликаты имен в вашем файле)
        if not review_text:
            return []

        # Безопасно парсим рейтинг, если поле пустое, используем 0 или None
        try:
            rating = int(rating_str)
        except ValueError:
            rating = 0 # Используем 0, если рейтинг отсутствует

        season = get_season(date_text)
        sentiment = 'positive' if rating >= 4 else 'negative'
        
        # Лемматизация текста
        doc = Doc(review_text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        
        lemmas = []
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            lemma = token.lemma
            if re.match(r'[а-яё]+', lemma) and lemma not in STOP_WORDS and len(lemma) > 2:
                lemmas.append((lemma, sentiment, season, rating))
        return lemmas
    except Exception as e:
        print(f"Критическая ошибка при разборе строки: {e} | Строка: {row_string.strip()}")
        return []

# --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ---

conn = sqlite3.connect('pharmacy_analysis_2026.db')
cursor = conn.cursor()
cursor.execute('DROP TABLE IF EXISTS word_data')
cursor.execute('CREATE TABLE word_data (word TEXT, sentiment TEXT, season TEXT, rating INTEGER)')

print("Начинаю обработку отзывов...")

# Предположим, отзывы лежат в файле 'reviews.txt'
# Если файла нет, можно заменить на чтение из списка
filename = 'yandex_reviews_clean.csv'
try:
    with open(filename, 'r', encoding='utf-8') as file:
        # Цикл while для построчного чтения
        line = file.readline()
        count = 0
        while line:
            if line.strip():  # Пропускаем пустые строки
                data = process_review(line)
                if data:
                    cursor.executemany('INSERT INTO word_data VALUES (?, ?, ?, ?)', data)
                    count += 1
            
            # Читаем следующую строку
            line = file.readline()
            
            # Выводим прогресс каждые 10 строк
            if count % 10 == 0 and count > 0:
                print(f"Обработано отзывов: {count}")

    conn.commit()
    print(f"Всего успешно обработано отзывов: {count}")

except FileNotFoundError:
    print("Файл " + filename + " не найден. Поместите отзывы в файл.")

# --- ПОДГОТОВКА ДЛЯ DATALENS ---
df = pd.read_sql_query("SELECT * FROM word_data", conn)
df.to_csv('datalens_final.csv', index=False, encoding='utf-8-sig')
print("Данные для Yandex DataLens сохранены в 'datalens_final.csv'")

conn.close()
