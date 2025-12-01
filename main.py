import os
import sys
import json
import re
import warnings
import time
from typing import List

from openai import OpenAI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
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

# Файлы для сохранения промежуточного прогресса
TEMP_OUTPUT_FILE = "temp_progress_book.epub"
CHECKPOINT_FILE = "checkpoint.json"

# Модель (используем flash для скорости, так как запросов будет много)
POE_MODEL = "gemini-2.5-flash-lite"

# Уменьшили размер батча для стабильности
BATCH_SIZE = 15

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
# ПРОМПТ (НЕ МЕНЯЕМ, КАК ПРОСИЛИ)
# ==========================================
SYSTEM_PROMPT = """
# ROLE & OBJECTIVE
Ты — Элитный Лингвист и Переводчик-Перфекционист (уровень носителя русского языка с PhD в филологии). Твоя задача — перевести **массив строк из JSON** с английского на русский язык, достигнув показателя качества **минимум 98%**.

# INPUT DATA
<source_text>
</source_text>

# TRANSLATION PROTOCOL (THE RECURSIVE LOOP)
Ты обязан использовать внутренний итеративный процесс (до 8 циклов), чтобы довести перевод до идеала. Не выводи промежуточные черновики пользователю, но строго следуй этому алгоритму "в уме":

1.  **Initial Draft:** Создай первый вариант перевода.
2.  **Critical Analysis (The 98% Bar):** Придирчиво оцени свой перевод по шкале от 0 до 100% на основе критериев:
    *   *Точность:* Нет ли искажений смысла?
    *   *Стиль:* Звучит ли это как естественная русская речь, а не как "перевод"? (Убраны ли кальки, канцеляризмы, пассивный залог).
    *   *Терминология:* Соблюдено ли единство терминов?
    *   *Техническое соответствие:* Соответствует ли вывод строгим правилам ниже?
3.  **Refinement:** Если оценка ниже 98%, перепиши текст, устраняя найденные недостатки.
4.  **Loop:** Повторяй шаги 2 и 3, пока не достигнешь 98%+ или пока не пройдешь 8 итераций.
5.  **Final Polish:** Сделай финальную вычитку на благозвучие.

# CRITICAL OUTPUT RULES
*   **ГЛАВНОЕ ПРАВИЛО:** Пользователь должен увидеть **ТОЛЬКО** финальный результат перевода в виде JSON-массива.
*   **СТРОГИЙ ФОРМАТ:** Верни ТОЛЬКО JSON-массив строк. Длина массива ответа ДОЛЖНА БЫТЬ РАВНА длине массива запроса.
*   **СОХРАНЕНИЕ СТРУКТУРЫ:** НЕ объединяй строки. Если строка пустая или состоит из пробелов — верни её как есть.
*   **КОД, ID, URL:** НЕ переводи код, ID, URL. Сохраняй знаки препинания и пробелы в начале/конце строки.
*   **ЗАПРЕЩЕНО:** Выводить текст вида "Итерация 1", "Оценка качества", "Вот перевод".
*   **ЗАПРЕЩЕНО:** Оставлять комментарии, примечания или свои мысли.
*   Твой ответ должен начинаться с квадратной скобки `[` и заканчиваться квадратной скобкой `]`.

# EXECUTION
Выполни протокол и выведи Идеальный Перевод:
"""

# ==========================================
# ФУНКЦИЯ ПАРСИНГА (Regex Fallback)
# ==========================================
def extract_strings_from_broken_json(text: str) -> List[str]:
    """
    Пытается извлечь строки из сломанного JSON с помощью Regex.
    Ищет паттерны "текст", учитывая экранированные кавычки.
    """
    # Этот паттерн ищет содержимое внутри двойных кавычек, игнорируя \" (экранированные)
    pattern = r'"((?:[^"\\]|\\.)*)"'
    matches = re.findall(pattern, text)
    
    # Regex возвращает сырые строки (с \\uXXXX и т.д.), нужно их "разэкранировать"
    decoded_matches = []
    for m in matches:
        try:
            # Оборачиваем в кавычки и парсим как JSON-строку, чтобы корректно обработать \n, \t, \"
            decoded_matches.append(json.loads(f'"{m}"'))
        except:
            # Если совсем всё плохо, берем как есть
            decoded_matches.append(m)
            
    return decoded_matches

# ==========================================
# ФУНКЦИЯ ПЕРЕВОДА
# ==========================================
def _call_api(texts: List[str]) -> List[str]:
    """Вызов API с защитой от битого JSON"""
    json_payload = json.dumps(texts, ensure_ascii=False)
    
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
        
        # Очистка от Markdown-оберток
        if content.startswith("```json"): content = content[7:]
        elif content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()

        # 1. Попытка стандартного парсинга
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # 2. Если упало — пробуем Regex Fallback
            # Это решает проблемы: Trailing comma, Missing comma, Unterminated string (иногда)
            extracted = extract_strings_from_broken_json(content)
            
            # Если Regex нашел столько же строк, сколько мы отправляли — считаем это победой
            if len(extracted) == len(texts):
                return extracted
            
            # Если не совпало количество, выбрасываем ошибку, чтобы сработал механизм Splitting
            raise ValueError(f"JSON broken and Regex mismatch (sent {len(texts)}, got {len(extracted)}). Raw err: {e}")

    except Exception as e:
        # Пробрасываем ошибку наверх, чтобы translate_batch_robust её поймал и разбил батч
        raise e

def translate_batch_robust(texts: List[str]) -> List[str]:
    """
    Рекурсивная функция. Если происходит ошибка API или несовпадение длины,
    разбивает список на две части и пробует снова.
    """
    if not texts:
        return []

    # Попытка перевести весь кусок
    try:
        translated = _call_api(texts)
        if len(translated) == len(texts):
            return translated
        else:
            print(f"    [Warn] Mismatch: sent {len(texts)}, got {len(translated)}. Splitting...")
    except Exception as e:
        # Ловим ошибки JSON и API
        err_msg = str(e).split('\n')[0] # Берем только первую строку ошибки для лога
        print(f"    [Error] {err_msg}. Splitting...")

    # === БАЗОВЫЙ СЛУЧАЙ РЕКУРСИИ ===
    # Если остался 1 элемент и он всё равно вызывает ошибку — мы НЕ можем его перевести.
    # Возвращаем оригинал, чтобы скрипт не зациклился и не упал.
    if len(texts) <= 1:
        print(f"    [Fail] Skipping problematic line (keeping original): {texts[0][:30]}...")
        return texts

    # === РАЗБИЕНИЕ (DIVIDE AND CONQUER) ===
    mid = len(texts) // 2
    left_part = texts[:mid]
    right_part = texts[mid:]

    # Рекурсивный вызов
    return translate_batch_robust(left_part) + translate_batch_robust(right_part)

# ==========================================
# ФУНКЦИИ СОХРАНЕНИЯ
# ==========================================
def save_checkpoint(book, last_chapter_idx):
    """Сохраняет текущее состояние книги и индекс последней обработанной главы."""
    try:
        # 1. Сохраняем промежуточный EPUB
        epub.write_epub(TEMP_OUTPUT_FILE, book, {})
        
        # 2. Сохраняем метаданные (индекс главы)
        data = {"last_chapter_idx": last_chapter_idx}
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        print(f"  [Checkpoint] Прогресс сохранен. Глава {last_chapter_idx} завершена.")
    except Exception as e:
        print(f"  [Error] Не удалось сохранить чекпоинт: {e}")

def load_checkpoint_info():
    """Читает файл чекпоинта и возвращает индекс последней главы или -1."""
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(TEMP_OUTPUT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_chapter_idx", -1)
        except Exception:
            return -1
    return -1

def clean_checkpoints():
    """Удаляет временные файлы после успешного завершения."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    if os.path.exists(TEMP_OUTPUT_FILE):
        os.remove(TEMP_OUTPUT_FILE)

# ==========================================
# MAIN
# ==========================================
def main():
    # 1. ПРОВЕРКА НАЛИЧИЯ ЧЕКПОИНТА
    last_processed_idx = load_checkpoint_info()
    
    if last_processed_idx >= 0:
        print(f"!!! ВОЗОБНОВЛЕНИЕ СЕССИИ !!!")
        print(f"Загрузка из: {TEMP_OUTPUT_FILE}")
        print(f"Пропуск глав: 1-{last_processed_idx + 1}")
        try:
            book = epub.read_epub(TEMP_OUTPUT_FILE)
        except Exception:
            print("Ошибка чекпоинта. Старт с нуля.")
            book = epub.read_epub(INPUT_FILE)
            last_processed_idx = -1
    else:
        print(f"Загрузка оригинала: {INPUT_FILE}")
        book = epub.read_epub(INPUT_FILE)

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    total_items = len(items)
    print(f"Всего документов в книге: {total_items}")

    try:
        for idx, item in enumerate(items):
            # Пропуск готовых глав
            if idx <= last_processed_idx:
                continue

            print(f"\n--- Глава {idx + 1}/{total_items}: {item.get_name()} ---")
            
            content = item.get_content()
            soup = BeautifulSoup(content, 'html.parser')

            text_nodes = []
            for node in soup.find_all(string=True):
                if not node.strip(): continue
                if isinstance(node, (BeautifulSoup, type(None))): continue
                if node.parent and node.parent.name in BLACKLIST_TAGS: continue
                text_nodes.append(node)

            total_nodes = len(text_nodes)
            print(f"  Фрагментов: {total_nodes}")
            
            if total_nodes > 0:
                # Используем tqdm-like вывод или просто счетчик
                for i in range(0, total_nodes, BATCH_SIZE):
                    batch_nodes = text_nodes[i : i + BATCH_SIZE]
                    batch_texts = [node.string for node in batch_nodes]
                    
                    print(f"  Батч {i//BATCH_SIZE + 1}/{(total_nodes//BATCH_SIZE) + 1}...", end="\r")
                    
                    # Запуск умного перевода
                    translated_texts = translate_batch_robust(batch_texts)
                    
                    for node, new_text in zip(batch_nodes, translated_texts):
                        if new_text and new_text.strip():
                            node.replace_with(new_text)

                # Сохраняем изменения в объект главы
                formatter = XMLFormatter()
                new_content = soup.encode('utf-8', formatter=formatter)
                item.set_content(new_content)
            
            # Чекпоинт после главы
            save_checkpoint(book, idx)
            print(f"  [OK] Глава сохранена.")

    except KeyboardInterrupt:
        print("\n\n!!! ПРИНУДИТЕЛЬНАЯ ОСТАНОВКА !!!")
        print("Текущий прогресс сохранен в чекпоинте.")
        sys.exit(0)

    print(f"\nФинализация: {OUTPUT_FILE}")
    epub.write_epub(OUTPUT_FILE, book, {})
    clean_checkpoints()
    print("Успешно завершено.")

if __name__ == "__main__":
    main()