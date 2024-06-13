import requests
import re
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever
from datasets import load_dataset, Dataset, DatasetDict


class SefariaDataCollector:
    def __init__(self, base_url='https://www.sefaria.org/api'):
        self.base_url = base_url

    def get_all_indices(self):
        """Fetch all text indices from the Sefaria API."""
        url = f'{self.base_url}/index'
        response = requests.get(url)
        indices = response.json()
        return [index['title'] for index in indices]

    def get_text_for_index(self, title):
        """Retrieve English texts from Sefaria API by title."""
        url = f'{self.base_url}/texts/{title}?context=0'
        try:
            response = requests.get(url)
            text_data = response.json()
            return text_data.get('text', '')  # Ensuring only English texts are returned
        except Exception as e:
            print(f"Failed to fetch data for {title}: {str(e)}")
            return ""

    def collect_texts(self):
        """Collect all texts available in the Sefaria database."""
        all_indices = self.get_all_indices()
        all_texts = {}
        for index in all_indices:
            text = self.get_text_for_index(index)
            if text:
                all_texts[index] = text
        return all_texts


class TextPreprocessor:
    def __init__(self):
        self.pattern = re.compile(r'[^a-zA-Z0-9\s]')

    def preprocess(self, text):
        """Clean text by removing non-alphanumeric characters."""
        return self.pattern.sub('', text)


class ChatbotBuilder:
    def __init__(self, all_texts):
        self.tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
        self.dataset = self.create_dataset(all_texts)
        self.model = None

    def create_dataset(self, texts):
        """Converts texts into a Hugging Face dataset."""
        data = {'text': list(texts.values())}
        return Dataset.from_dict(data)

    def build_model(self):
        """Initializes and builds the RAG model."""
        retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', dataset=self.dataset)
        self.model = RagTokenForGeneration.from_pretrained('facebook/rag-token-nq', retriever=retriever)


if __name__ == "__main__":
    collector = SefariaDataCollector()
    all_texts = collector.collect_texts()

    preprocessor = TextPreprocessor()
    preprocessed_texts = {title: preprocessor.preprocess(text) for title, text in all_texts.items()}

    chatbot = ChatbotBuilder(preprocessed_texts)
    chatbot.build_model()

    print("Chatbot built successfully with RAG model!")
