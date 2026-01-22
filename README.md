# AI EPUB Translator

A tool for automated translation of EPUB books into Russian using the Poe API (Gemini 2.5 Flash Lite model).

## Features

*   Translation of EPUB books from English to Russian
*   Preservation of the original HTML document structure
*   Checkpoint system for resuming interrupted translations
*   Batch text processing with automatic splitting on errors
*   Smart prompt with iterative translation quality improvement

## Requirements

*   Python 3.13.7+
*   A Poe API key

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/andsergey13/ai-epub-translator.git
    cd ai-epub-translator
    ```

2.  Install the dependencies:
    ```bash
    pip install beautifulsoup4 python-dotenv ebooklib lxml openai
    ```

3.  Create a `.env` file in the project root and add your API key:
    ```
    API_POE_KEY=your_poe_api_key
    ```

## Usage

1.  Place the source EPUB file in the project directory under the name `original_book.epub`

2.  Run the script:
    ```bash
    python main.py
    ```

3.  The translated book will be saved as `translate_book_RU.epub`

## Configuration

Parameters are configured at the beginning of the `main.py` file:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `INPUT_FILE` | `original_book.epub` | Source file name |
| `OUTPUT_FILE` | `translate_book_RU.epub` | Output file name |
| `POE_MODEL` | `gemini-2.5-flash-lite` | Model to use |
| `BATCH_SIZE` | `15` | Number of text fragments per request |

## How It Works

1.  **EPUB Parsing:** Extracts HTML documents from the book using `ebooklib`.
2.  **Text Extraction:** Identifies text nodes via `BeautifulSoup`, excluding technical tags (`script`, `style`, `code`, `pre`, etc.).
3.  **Batch Translation:** Sends text for translation in batches formatted as JSON arrays.
4.  **Error Handling:** Recursively splits a batch in half if the response length does not match the original.
5.  **Checkpoints:** Saves progress after each chapter to `checkpoint.json` and `temp_progress_book.epub`.

## Resuming Translation

If the process is interrupted (Ctrl+C or an error), progress is saved automatically. Running the script again will resume translation from the last saved chapter.

To start over from the beginning, delete these files:
*   `checkpoint.json`
*   `temp_progress_book.epub`

## Known Limitations

*   Translation Accuracy: 88–92% (target is 95%)
*   Possible formatting changes: centering, font type, font size