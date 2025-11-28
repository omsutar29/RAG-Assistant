# 📺 YouTube Playlist RAG Assistant

**Turn any YouTube Course into a Searchable Knowledge Base.**

This project is a Retrieval-Augmented Generation (RAG) system designed to optimize the learning experience from video courses. Instead of scrubbing through hours of video to find specific concepts, users can simply "chat" with the playlist.

*Example:* You upload a 100-video "Python for Beginners" course. You ask: *"Where are Python functions taught?"* The assistant replies: *"This topic is covered in **Video #6 (Introduction to Methods)** at timestamp **12:45**."*

---

## ⚠️ Important: Run on Google Colab

**This project requires significant GPU computing power.** It utilizes **OpenAI's Whisper (Large-v2)** for transcription and **Qwen-1.5B** for reasoning. Running this locally requires a high-end GPU with substantial VRAM.

**For the best experience, please run these notebooks in Google Colab using a GPU Runtime (T4 or better).**

---

## 🚀 How It Works (The Pipeline)

The project consists of 5 sequential notebooks that process raw video data into a queryable AI agent.

### 1. Ingestion (`01_download_playlist_ytdlp.ipynb`)
Uses `yt-dlp` to download a full YouTube playlist.
* **Action:** Paste your playlist URL.
* **Output:** Downloads videos and automatically formats filenames (e.g., `Tut_1 - Title.mp4`).

### 2. Audio Extraction (`02_video_to_audio.ipynb`)
Uses `ffmpeg` to strip video data and retain high-quality audio.
* **Action:** Run the notebook to convert `.mp4` video files to `.mp3`.
* **Output:** An optimized `audios/` folder ready for transcription.

### 3. Transcription (`03_speech_to_text.ipynb`)
The heavy lifter. Uses **OpenAI Whisper (Large-v2)** to transcribe audio with human-level accuracy.
* **Action:** Run the transcription loop.
* **Output:** Generates detailed JSON files containing text chunks and accurate timestamps.

### 4. Vector Embedding (`04_creating_embeddings_for_jsons.ipynb`)
Converts text data into mathematical vectors using **BAAI/bge-m3**.
* **Action:** Processes the JSON transcripts.
* **Output:** A serialized `embeddings.joblib` file (the "brain" of the RAG system).

### 5. The Assistant (`05_process_incoming_query.ipynb`)
The interface. Uses **Qwen-1.5B-Instruct** (LLM) and Cosine Similarity.
* **Action:** Load the embeddings and ask your question.
* **Result:** The AI retrieves the exact video segments and timestamps relevant to your query.

---

## 🛠️ Usage Guide

To use this assistant, follow these steps in order:

1.  **Clone/Download** this repository.
2.  **Upload the Notebooks** to your [Google Colab](https://colab.research.google.com/).
3.  **Enable GPU:** In Colab, go to `Runtime` > `Change runtime type` > Select `T4 GPU`.
4.  **Run Sequentially:**
    * Run Notebook `01`. When prompted, paste the link to the YouTube playlist you want to learn from.
    * Run Notebooks `02`, `03`, and `04` to process the data.
    * Run Notebook `05`. When prompted `Ask a Question:`, type your query (e.g., *"What is the difference between list and tuple?"*).

---

## ⚙️ Customization: Using a Different LLM

By default, this project uses **Qwen2-1.5B-Instruct** because it offers an excellent balance of speed and intelligence for free-tier Colab GPUs. However, you can easily swap this for other Hugging Face models (e.g., **Microsoft Phi-3**, **Google Gemma**, or **TinyLlama**).

To change the LLM:

1.  Open **`05_process_incoming_query.ipynb`**.
2.  Locate the cell marked `# Load LLM once`.
3.  Find these lines of code:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        device_map="auto"
    )
    ```
4.  Replace `"Qwen/Qwen2-1.5B-Instruct"` with the Model ID of your choice from Hugging Face.

**Recommended alternatives for Colab (T4 GPU):**
* `microsoft/Phi-3-mini-4k-instruct` (High performance, small size)
* `google/gemma-2b-it` (Google's lightweight open model)
* `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Ultra-fast, lower memory usage)

*Note: If you choose a model larger than 7 Billion parameters (e.g., Llama-3-8B), you may run out of memory (OOM) on the free Colab tier.*

---

## 🤖 Tech Stack

* **Ingestion:** `yt-dlp`
* **Audio Processing:** `ffmpeg`
* **ASR (Speech-to-Text):** `openai-whisper` (Large-v2 model)
* **Embeddings:** `SentenceTransformers` (Model: `BAAI/bge-m3`)
* **Vector Search:** `scikit-learn` (Cosine Similarity)
* **LLM:** `transformers` (Model: `Qwen/Qwen2-1.5B-Instruct`)

---

## 📝 Requirements

If you are running this locally (not recommended without GPU), you will need the dependencies listed in `requirements.txt`. Note that the Notebooks contain cells to automatically install these dependencies in the Colab environment.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)