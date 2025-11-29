import os
import sys
import json
import warnings
import time
from typing import List

from openai import OpenAI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
# Исправление ошибки KeyError: 'xml' - импортируем класс явно
from bs4.formatter import XMLFormatter
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Отключаем ворнинги
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
INPUT_FILE = "original_book.epub"
OUTPUT_FILE = "translate_book_RU.epub"

# Модель (используем flash для скорости, так как запросов будет много)
POE_MODEL = "gemini-2.5-flash-lite"

# Размер батча (количество текстовых фрагментов за раз)
# Для этого метода лучше брать побольше, так как фрагменты могут быть короткими
BATCH_SIZE = 30

# Теги, содержимое которых НЕЛЬЗЯ переводить
BLACKLIST_TAGS = {
    'script', 'style', 'pre', 'code', 'kbd', 'samp', 
    'var', 'head', 'title', 'meta', 'link'
}

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
# ПРОМПТ
# ==========================================
SYSTEM_PROMPT = """
# ROLE & OBJECTIVE
Ты — Элитный Лингвист и Переводчик-Перфекционист (уровень носителя русского языка с PhD в филологии). Твоя задача — перевести предоставленный список текстовых фрагментов с английского на русский язык, достигнув показателя качества **минимум 98%**.

# INPUT DATA
<source_text>
</source_text>

# TRANSLATION PROTOCOL (THE RECURSIVE LOOP)
Ты обязан использовать внутренний итеративный процесс (до 8 циклов), чтобы довести перевод до идеала. Не выводи промежуточные черновики пользователю, но строго следуй этому алгоритму "в уме":

1.  **Initial Draft:** Создай первый вариант перевода В ФОРМАТЕ JSON-МАССИВА СТРОК.
2.  **Critical Analysis (The 98% Bar):** Придирчиво оцени свой перевод по шкале от 0 до 100% на основе критериев:
    *   *Точность:* Нет ли искажений смысла?
    *   *Стиль:* Звучит ли это как естественная русская речь, а не как "перевод"? (Убраны ли кальки, канцеляризмы, пассивный залог).
    *   *Терминология:* Соблюдено ли единство терминов? НЕ ПЕРЕВОДИ технические термины, если они выглядят как код или ID.
    *   *Формат:* Сохранена ли точная длина массива? Сохранены ли стили пунктуации и пробелов?
3.  **Refinement:** Если оценка ниже 98%, перепиши текст, устраняя найденные недостатки.
4.  **Loop:** Повторяй шаги 2 и 3, пока не достигнешь 98%+ или пока не пройдешь 8 итераций.
5.  **Final Polish:** Сделай финальную вычитку на благозвучие.

# CRITICAL OUTPUT RULES
*   **ГЛАВНОЕ ПРАВИЛО:** Пользователь должен увидеть **ТОЛЬКО** финальный результат перевода — JSON-МАССИВ СТРОК.
*   **ВЕРНИ ТОЛЬКО JSON-МАССИВ СТРОК.**
*   **СОХРАНИ ТОЧНУЮ ДЛИНУ МАССИВА.**
*   **НЕ ПЕРЕВОДИ ТЕХНИЧЕСКИЕ ТЕРМИНЫ,** если они выглядят как код или ID.
*   **СОХРАНИ СТИЛИ ПУНКТУАЦИИ И ПРОБЕЛОВ** (если строка начинается с пробела — сохрани его).
*   **КОНТЕКСТ:** Это книга (художественная, научная и другие разновидности).
*   **ЗАПРЕЩЕНО:** Выводить текст вида "Итерация 1", "Оценка качества", "Вот перевод".
*   **ЗАПРЕЩЕНО:** Оставлять комментарии, примечания или свои мысли.
*   Твой ответ должен начинаться с квадратной скобки `[` и заканчиваться квадратной скобкой `]`.

# EXECUTION
Выполни протокол и выведи Идеальный Перевод В ФОРМАТЕ JSON-МАССИВА:
"""

# ==========================================
# ФУНКЦИЯ ПЕРЕВОДА
# ==========================================
def translate_batch(texts: List[str]) -> List[str]:
    """Отправляет список строк на перевод и возвращает список переводов"""
    if not texts:
        return []
    
    # Очистка от пустых запросов, чтобы не тратить токены (хотя мы фильтруем их ранее)
    json_payload = json.dumps(texts, ensure_ascii=False)
    
    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=POE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json_payload}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            
            # Очистка от markdown форматирования, если модель его добавила
            if content.startswith("```json"): content = content[7:]
            if content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            
            translated = json.loads(content)
            
            if len(translated) != len(texts):
                print(f"  [Warn] Mismatch length: sent {len(texts)}, got {len(translated)}. Retrying...")
                continue
                
            return translated
            
        except Exception as e:
            print(f"  [Error] Attempt {attempt+1}/{retries}: {e}")
            time.sleep(2)
    
    # Если все попытки провалились, возвращаем оригинал
    print("  [Fail] Batch translation failed. Returning originals.")
    return texts

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"Загрузка книги: {INPUT_FILE}")
    try:
        book = epub.read_epub(INPUT_FILE)
    except Exception as e:
        print(f"Ошибка открытия файла: {e}")
        sys.exit(1)

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    total_items = len(items)
    
    print(f"Найдено документов: {total_items}")

    for idx, item in enumerate(items):
        print(f"\n--- Обработка главы {idx + 1}/{total_items}: {item.get_name()} ---")
        
        # Используем html.parser
        content = item.get_content()
        soup = BeautifulSoup(content, 'html.parser')

        # 1. СБОР ТЕКСТОВЫХ УЗЛОВ (TEXT NODES)
        # Мы ищем все NavigableString, которые не являются пробелами и не внутри запрещенных тегов
        text_nodes = []
        
        for node in soup.find_all(string=True):
            # Пропускаем, если это просто перенос строки или пробелы
            if not node.strip():
                continue
            
            # Пропускаем комментарии и спец. директивы
            if isinstance(node, (BeautifulSoup, type(None))): 
                continue
                
            # Проверяем родителей на наличие в черном списке (стили, скрипты)
            if node.parent and node.parent.name in BLACKLIST_TAGS:
                continue
                
            text_nodes.append(node)

        total_nodes = len(text_nodes)
        print(f"  Найдено текстовых фрагментов: {total_nodes}")
        
        if total_nodes == 0:
            continue

        # 2. ПАКЕТНЫЙ ПЕРЕВОД
        # Разбиваем на батчи и переводим
        for i in range(0, total_nodes, BATCH_SIZE):
            batch_nodes = text_nodes[i : i + BATCH_SIZE]
            batch_texts = [node.string for node in batch_nodes]
            
            print(f"  Перевод батча {i//BATCH_SIZE + 1}/{(total_nodes//BATCH_SIZE) + 1} ({len(batch_texts)} строк)...", end="\r")
            
            translated_texts = translate_batch(batch_texts)
            
            # 3. ЗАМЕНА ТЕКСТА В DOM
            # Самая важная часть: мы меняем содержимое узла на месте
            for node, new_text in zip(batch_nodes, translated_texts):
                if new_text and new_text.strip():
                    # replace_with заменяет узел в дереве
                    node.replace_with(new_text)

        print(f"\n  Глава обработана.")

        # 4. СОХРАНЕНИЕ
        # Используем XMLFormatter явно, чтобы избежать KeyError: 'xml'
        # Это гарантирует валидный XHTML для EPUB (закрытые теги <br/>, <link/> и т.д.)
        formatter = XMLFormatter()
        new_content = soup.encode('utf-8', formatter=formatter)
        item.set_content(new_content)

    print(f"\nСохранение итогового файла: {OUTPUT_FILE}")
    epub.write_epub(OUTPUT_FILE, book, {})
    print("Готово! Структура полностью сохранена.")

if __name__ == "__main__":
    main()