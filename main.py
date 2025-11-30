import os
import sys
import json
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
# ФУНКЦИЯ ПЕРЕВОДА (С РЕКУРСИВНЫМ РАЗБИЕНИЕМ)
# ==========================================
def _call_api(texts: List[str]) -> List[str]:
    """Базовый вызов API"""
    json_payload = json.dumps(texts, ensure_ascii=False)
    
    response = client.chat.completions.create(
        model=POE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json_payload}
        ],
        temperature=0.3
    )
    content = response.choices[0].message.content.strip()
    
    if content.startswith("```json"): content = content[7:]
    if content.startswith("```"): content = content[3:]
    if content.endswith("```"): content = content[:-3]
    
    return json.loads(content)

def translate_batch_robust(texts: List[str]) -> List[str]:
    """
    Умная функция перевода.
    Если длина ответа не совпадает, разбивает батч на части и пробует снова.
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
        print(f"    [Error] API Error: {e}. Splitting...")

    # Если мы здесь, значит произошла ошибка или несовпадение длины.
    # Если остался всего 1 элемент и он не перевелся — возвращаем оригинал (чтобы не зациклиться)
    if len(texts) <= 1:
        print(f"    [Fail] Skipping problematic line: {texts[0][:20]}...")
        return texts

    # Разбиваем батч пополам (Divide and Conquer)
    mid = len(texts) // 2
    left_part = texts[:mid]
    right_part = texts[mid:]

    # Рекурсивно переводим половинки
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
                for i in range(0, total_nodes, BATCH_SIZE):
                    batch_nodes = text_nodes[i : i + BATCH_SIZE]
                    batch_texts = [node.string for node in batch_nodes]
                    
                    print(f"  Батч {i//BATCH_SIZE + 1}/{(total_nodes//BATCH_SIZE) + 1}...", end="\r")
                    
                    # Используем новую умную функцию
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
        print("Вы можете перезапустить скрипт позже, он продолжит работу.")
        sys.exit(0)

    print(f"\nФинализация: {OUTPUT_FILE}")
    epub.write_epub(OUTPUT_FILE, book, {})
    clean_checkpoints()
    print("Успешно завершено.")

if __name__ == "__main__":
    main()