from os import environ
import sys
import re
import warnings
from copy import copy

from openai import OpenAI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag

from dotenv import load_dotenv
load_dotenv()

# Отключаем шумные ворнинги
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
INPUT_FILE = "The_story_of_an_hour_short_story_Kate_Chopin.epub"
OUTPUT_FILE = "The_story_of_an_hour_short_story_Kate_Chopin_RU.epub"
POE_MODEL = "gemini-2.5-flash-lite"

# Теги, которые мы считаем "контейнерами" текста (блочные)
BLOCK_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'caption', 'td', 'th', 'div']
# Теги, которые мы будем превращать в маркеры <tN> (инлайновые)
INLINE_TAGS = ['a', 'b', 'strong', 'i', 'em', 'span', 'small', 'sub', 'sup', 'code']

# ==========================================
# КЛИЕНТ POE (OpenAI Compatible)
# ==========================================
# Согласно документации: https://creator.poe.com/docs/external-applications/openai-compatible-api
client = OpenAI(
    api_key=environ.get("API_POE_KEY"),
    base_url="https://api.poe.com/v1"
)

def translate_text_with_poe(text_segment):
    """
    Отправляет текст с маркерами <t0>...</t0> в Poe.
    """
    system_prompt = """
    Ты профессиональный переводчик литературы. Твоя задача — перевести текст с английского на русский.
    
    ИНСТРУКЦИЯ ПО СТРУКТУРЕ:
    1. Текст содержит технические маркеры вида <t0>текст</t0>, <t1>текст</t1>.
    2. Эти маркеры обозначают форматирование (ссылки, жирный шрифт).
    3. ТЫ ОБЯЗАН сохранить эти маркеры в переводе на соответствующих местах.
    4. ТЫ ОБЯЗАН перевести текст ВНУТРИ маркеров.
    
    Пример:
    Input: Click <t0>here</t0> to go.
    Output: Нажми <t0>сюда</t0>, чтобы перейти.
    
    Не добавляй никаких вступлений, только переведенный текст.
    """

    try:
        response = client.chat.completions.create(
            model=POE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_segment}
            ],
            temperature=0.3 # Низкая температура для точности сохранения тегов
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n[API Error] {e}")
        return text_segment # Возвращаем оригинал в случае ошибки

# ==========================================
# ЛОГИКА "SKELETON" (МАРКЕРЫ)
# ==========================================

class TagRegistry:
    """Хранилище для оригинальных тегов"""
    def __init__(self):
        self.registry = {}
        self.counter = 0

    def register(self, tag):
        """Заменяет реальный тег на маркер и сохраняет оригинал"""
        key = f"t{self.counter}"
        self.counter += 1
        self.registry[key] = copy(tag) # Сохраняем копию тега с атрибутами
        return key

    def get_original(self, key):
        return self.registry.get(key)

def process_block(block_tag):
    """
    Берет HTML блок (например <p>), превращает инлайн теги в <tN>,
    переводит текст, восстанавливает теги.
    """
    # 1. Проверка: есть ли текст?
    if not block_tag.get_text(strip=True):
        return

    registry = TagRegistry()
    
    # 2. Создаем "Скелет": заменяем инлайн теги на <tN>
    # Мы работаем с копией, чтобы не ломать итерацию, но модифицируем оригинал
    # Для упрощения создадим временную строку-представление
    
    # Стратегия: Проходим по содержимому. Если это NavigableString - берем текст.
    # Если Tag - регистрируем, берем его внутренний текст, оборачиваем в <tN>.
    
    skeleton_parts = []
    
    for child in block_tag.contents:
        if isinstance(child, NavigableString):
            skeleton_parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in INLINE_TAGS:
                # Рекурсивно получаем текст внутри тега (на случай вложенности, 
                # хотя для простоты считаем 1 уровень достаточным для большинства книг)
                inner_text = child.get_text() 
                marker = registry.register(child)
                skeleton_parts.append(f"<{marker}>{inner_text}</{marker}>")
            else:
                # Если встретили что-то странное (например img), оставляем как есть (placeholder)
                skeleton_parts.append(str(child))
    
    skeleton_text = "".join(skeleton_parts)
    
    # Если текст пустой или состоит только из пробелов
    if not skeleton_text.strip():
        return

    # 3. Перевод через Poe
    translated_text = translate_text_with_poe(skeleton_text)
    
    # 4. Восстановление (Re-hydration)
    # Нам нужно распарсить переведенный текст, найти <tN> и заменить их обратно на реальные теги
    
    # Парсим полученный от LLM фрагмент как HTML
    soup_fragment = BeautifulSoup(translated_text, 'html.parser')
    
    new_contents = []
    
    for element in soup_fragment.contents:
        if isinstance(element, NavigableString):
            new_contents.append(element)
        elif isinstance(element, Tag):
            # Проверяем, является ли это нашим маркером tN
            if re.match(r't\d+', element.name):
                original_tag = registry.get_original(element.name)
                if original_tag:
                    # Клонируем оригинальный тег (чтобы сохранить href, class и т.д.)
                    restored_tag = copy(original_tag)
                    # Вставляем внутрь переведенный текст
                    restored_tag.string = element.get_text()
                    new_contents.append(restored_tag)
                else:
                    # Если LLM выдумала несуществующий тег, просто возвращаем текст
                    new_contents.append(NavigableString(element.get_text()))
            else:
                # Какой-то другой тег, который LLM вернула (бывает редко)
                new_contents.append(element)
                
    # 5. Обновляем исходный блок
    block_tag.clear()
    for item in new_contents:
        block_tag.append(item)

# ==========================================
# ОСНОВНОЙ ЦИКЛ
# ==========================================

def main():
    if not environ.get("API_POE_KEY"):
        print("Ошибка: Не задана переменная окружения API_POE_KEY")
        sys.exit(1)

    print(f"Загрузка книги: {INPUT_FILE}")
    try:
        book = epub.read_epub(INPUT_FILE)
    except FileNotFoundError:
        print("Файл не найден!")
        sys.exit(1)

    total_items = 0
    processed_items = 0

    # Считаем документы для прогресса
    docs = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
    total_docs = len(docs)

    print(f"Найдено глав: {total_docs}. Начинаем перевод...")

    for item in docs:
        processed_items += 1
        print(f"Processing chapter {processed_items}/{total_docs}: {item.get_name()}")
        
        content = item.get_content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Находим все блочные элементы
        blocks = soup.find_all(BLOCK_TAGS)
        
        # Проходим по блокам
        for i, block in enumerate(blocks):
            # Простейший вывод прогресса внутри главы
            if i % 10 == 0:
                sys.stdout.write(f"\r  Block {i}/{len(blocks)}")
                sys.stdout.flush()
            
            process_block(block)
            
        print("") # Newline
        
        # Сохраняем измененный контент обратно в книгу
        item.set_content(soup.encode(formatter="html"))

    print(f"Сохранение результата в {OUTPUT_FILE}...")
    epub.write_epub(OUTPUT_FILE, book, {})
    print("Готово!")

if __name__ == "__main__":
    main()