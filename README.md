# Laptop Recommender Chatbot

## 🚀 Project Overview

This project is a Laptop Recommender Chatbot built using Python, LangChain, Hugging Face Sentence Transformers, and Gradio. The chatbot helps users by:

* Recommending laptops based on user queries and specified preferences (price, RAM, features, etc.).
* Extracting key features of laptops (RAM, storage, battery, etc.).
* Comparing multiple laptops based on user-specified features.
* Summarizing conversation history for user convenience.

## 📌 Key Features

* *Laptop Recommendations:* Provides personalized laptop suggestions based on user queries.
* *Feature Extraction:* Extracts key laptop specifications using natural language processing.
* *Product Comparison:* Compares multiple laptops on features like price, RAM, storage, battery life, and more.
* *Conversation Summarization:* Generates a summary of the chat history using LangChain’s summarization model.

## 💡 How It Works

* Laptops are loaded from a CSV file (first_500_records.csv), and each laptop is transformed into a document with key metadata (name, price, specs, etc.).
* Hugging Face SentenceTransformer (all-MiniLM-L6-v2) is used to generate embeddings for these laptop descriptions.
* FAISS is used as the vector store for efficient similarity search.
* A LangChain RetrievalQA setup is used to handle user queries with the help of the Groq API for LLM responses.
* Gradio provides an interactive chat UI for users to interact with the chatbot.

## 📝 Prerequisites

* Python 3.8+
* pip (Python package installer)
* An API key for the Groq API (stored in a .env file)

## 📁 Project Structure

* first_500_records.csv: The dataset containing laptop details.
* chatbot.py: Main Python script containing all logic (loading data, embedding creation, recommendation, feature extraction, comparison, and UI).
* .env: Environment file storing API keys (GROQ\_API\_KEY).

## 🛠 Installation

1. Clone this repository:

   bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   
2. Create a virtual environment:

   bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   
3. Install the required libraries:

   bash
   pip install -r requirements.txt
   
4. Create a .env file and add your Groq API key:

   bash
   GROQ_API_KEY=your_groq_api_key_here
   

## 🚀 Running the Chatbot

bash
python chatbot.py


## 📌 Usage Examples

* "Recommend laptops under ₹50,000"
* "Compare laptops with 16GB RAM"
* "Show laptops with i7 processor"
* "Summarize our conversation"

## 🌐 Built With

* Python
* LangChain
* Hugging Face Sentence Transformers
* FAISS (Facebook AI Similarity Search)
* Gradio

## 📄 License

This project is open-source. Feel free to modify and enhance it as needed.

## 💬 Acknowledgments

Special thanks to the developers of LangChain, Hugging Face, and Gradio for their amazing tools.

