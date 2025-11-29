import os
import sys
import re
import json
import warnings
from copy import copy
from typing import List, Dict

from openai import OpenAI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Отключаем шумные ворнинги
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
INPUT_FILE = "The_story_of_an_hour_short_story_Kate_Chopin.epub"
OUTPUT_FILE = "The_story_of_an_hour_short_story_Kate_Chopin_RU.epub"

# Используем модель, которую ты указал. 
# Flash-lite быстрая, но для литературного перевода лучше Sonnet или GPT-4o.
# Если качество будет низким, попробуй сменить на "Claude-3.5-Sonnet".
POE_MODEL = "gemini-2.5-flash-lite" 

# Размер батча (количество текстовых блоков за 1 запрос)
# Для маленькой книги можно ставить больше, но 15-20 - безопасный оптимум для контекста
BATCH_SIZE = 20 

# Теги, которые содержат текст. УБРАЛ 'div', чтобы не ломать верстку контейнеров.
BLOCK_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'caption', 'td', 'th']

# Инлайн теги для маскировки
INLINE_TAGS = ['a', 'b', 'strong', 'i', 'em', 'span', 'small', 'sub', 'sup', 'code']

# ==========================================
# КЛИЕНТ POE
# ==========================================
api_key = os.environ.get("API_POE_KEY")
if not api_key:
    print("Ошибка: Не задана переменная окружения API_POE_KEY")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.poe.com/v1"
)

# ==========================================
# ЛОГИКА "SKELETON" (МАРКЕРЫ)
# ==========================================

class TagRegistry:
    """Хранилище для оригинальных тегов, чтобы восстановить их после перевода"""
    def __init__(self):
        self.registry = {}
        self.counter = 0

    def register(self, tag):
        """Заменяет реальный тег на маркер <tN>"""
        key = f"t{self.counter}"
        self.counter += 1
        self.registry[key] = copy(tag)
        return key

    def get_original(self, key):
        return self.registry.get(key)

def skeletonize_block(block_tag, registry: TagRegistry) -> str:
    """Превращает HTML блок в текст с маркерами <tN>"""
    skeleton_parts = []
    
    for child in block_tag.contents:
        if isinstance(child, NavigableString):
            skeleton_parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in INLINE_TAGS:
                inner_text = child.get_text()
                # Если внутри тега пусто или только пробелы, сохраняем как есть, не переводим
                if not inner_text.strip():
                     skeleton_parts.append(str(child))
                     continue
                
                marker = registry.register(child)
                skeleton_parts.append(f"<{marker}>{inner_text}</{marker}>")
            else:
                # Неизвестный тег (например br или img) просто оставляем текстом
                skeleton_parts.append(str(child))
    
    return "".join(skeleton_parts)

def restore_block(translated_text: str, original_block: Tag, registry: TagRegistry):
    """Восстанавливает HTML из переведенного текста с маркерами"""
    # Очищаем старый контент, но САМ ТЕГ (например <p class="center">) остается на месте
    original_block.clear()
    
    # Парсим фрагмент
    soup_fragment = BeautifulSoup(translated_text, 'html.parser')
    
    new_contents = []
    for element in soup_fragment.contents:
        if isinstance(element, NavigableString):
            new_contents.append(element)
        elif isinstance(element, Tag):
            if re.match(r't\d+', element.name):
                original_tag = registry.get_original(element.name)
                if original_tag:
                    restored_tag = copy(original_tag)
                    restored_tag.string = element.get_text()
                    new_contents.append(restored_tag)
                else:
                    # Если LLM галлюцинировала тег
                    new_contents.append(NavigableString(element.get_text()))
            else:
                new_contents.append(element)
    
    for item in new_contents:
        original_block.append(item)

# ==========================================
# ЛОГИКА ПЕРЕВОДА (BATCHING)
# ==========================================

def translate_batch(texts: List[str]) -> List[str]:
    """Отправляет список строк в Poe и получает список переводов"""
    
    # Твой промпт "Элитного Лингвиста" + Техническая инструкция JSON
    system_prompt = """
# ROLE & OBJECTIVE
Ты — Элитный Лингвист и Переводчик-Перфекционист (уровень носителя русского языка с PhD в филологии). Твоя задача — перевести предоставленный список текстовых фрагментов на русский язык, достигнув показателя качества **минимум 98%**.

# INPUT FORMAT
Ты получишь JSON-список строк. Некоторые строки содержат технические маркеры вида <t0>...</t0>.
Пример: ["Hello <t0>world</t0>", "Chapter 1"]

# CRITICAL TECHNICAL RULES
1. **FORMAT:** Твой ответ должен быть **СТРОГО валидным JSON-списком** строк. Никакого Markdown, никаких ```json``` оберток. Просто `["перевод 1", "перевод 2"]`.
2. **TAGS:** Маркеры <tN>...</tN> (где N - число) — это святыня.
   - Ты ОБЯЗАН сохранить их.
   - Ты ОБЯЗАН перевести текст ВНУТРИ них.
   - Ты НЕ ИМЕЕШЬ ПРАВА менять число N внутри тега.
   - Пример: "Click <t5>here</t5>" -> "Нажми <t5>сюда</t5>".
3. **LENGTH:** Количество элементов в выходном списке должно В ТОЧНОСТИ совпадать с входным.

# TRANSLATION PROTOCOL (THE RECURSIVE LOOP)
Используй свой внутренний итеративный процесс (до 8 циклов), чтобы довести перевод до идеала:
1.  **Accuracy:** Нет ли искажений смысла?
2.  **Style:** Звучит ли это как естественная русская речь? (Убраны ли кальки, канцеляризмы).
3.  **Terminology:** Соблюдено ли единство?

Выведи ТОЛЬКО финальный JSON-список.
"""

    user_content = json.dumps(texts, ensure_ascii=False)

    try:
        response = client.chat.completions.create(
            model=POE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        # Очистка от markdown-блоков, если модель их добавила вопреки инструкции
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        translated_texts = json.loads(content)
        
        if len(translated_texts) != len(texts):
            print(f"\n[Warning] Mismatch length! Sent {len(texts)}, got {len(translated_texts)}. Trying to align...")
            # Если длины не совпадают, это авария. Возвращаем оригинал для безопасности.
            return texts
            
        return translated_texts

    except json.JSONDecodeError:
        print(f"\n[Error] Model returned invalid JSON. Returning originals.")
        print(f"Raw output start: {content[:100]}...")
        return texts
    except Exception as e:
        print(f"\n[API Error] {e}")
        return texts

# ==========================================
# ОСНОВНОЙ ЦИКЛ
# ==========================================

def main():
    print(f"Загрузка книги: {INPUT_FILE}")
    try:
        book = epub.read_epub(INPUT_FILE)
    except FileNotFoundError:
        print("Файл не найден!")
        sys.exit(1)

    docs = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
    total_docs = len(docs)
    
    print(f"Найдено глав: {total_docs}. Используем модель: {POE_MODEL}")

    # Глобальный реестр тегов для всей книги (или можно создавать на главу, но лучше глобально для уникальности ID)
    # Но для простоты сделаем реестр на каждый батч или главу. 
    # Лучше на главу, чтобы не путаться.
    
    for doc_idx, item in enumerate(docs):
        print(f"\nProcessing chapter {doc_idx + 1}/{total_docs}: {item.get_name()}")
        
        content = item.get_content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Находим блоки. ВАЖНО: recursive=True по умолчанию в find_all, но нам нужно
        # убедиться, что мы не берем вложенные блоки дважды.
        # Стратегия: берем все P, H1..H6, LI. Они редко вложены друг в друга валидно.
        blocks = soup.find_all(BLOCK_TAGS)
        
        # Фильтруем пустые блоки
        valid_blocks = [b for b in blocks if b.get_text(strip=True)]
        
        if not valid_blocks:
            continue

        # Подготовка батчей
        current_batch_texts = []
        current_batch_blocks = [] # Ссылки на объекты BeautifulSoup
        current_registry = TagRegistry() # Реестр для текущей главы
        
        print(f"  Found {len(valid_blocks)} text blocks. Batching...")

        for i, block in enumerate(valid_blocks):
            # Скелетируем текст (заменяем <a> на <t0>)
            text_skeleton = skeletonize_block(block, current_registry)
            
            current_batch_texts.append(text_skeleton)
            current_batch_blocks.append(block)
            
            # Если батч заполнен или это последний блок
            if len(current_batch_texts) >= BATCH_SIZE or i == len(valid_blocks) - 1:
                # Отправляем в Poe
                sys.stdout.write(f"\r  Translating batch {i // BATCH_SIZE + 1}...")
                sys.stdout.flush()
                
                translated_batch = translate_batch(current_batch_texts)
                
                # Применяем переводы обратно к блокам
                for block_ref, trans_text in zip(current_batch_blocks, translated_batch):
                    restore_block(trans_text, block_ref, current_registry)
                
                # Очищаем батч
                current_batch_texts = []
                current_batch_blocks = []

        # Сохраняем главу
        item.set_content(soup.encode(formatter="html"))

    print(f"\nСохранение результата в {OUTPUT_FILE}...")
    epub.write_epub(OUTPUT_FILE, book, {})
    print("Готово! Проверь структуру и качество.")

if __name__ == "__main__":
    main()