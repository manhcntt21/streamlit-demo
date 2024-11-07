from flask import Flask, request, jsonify
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from transformers import BitsAndBytesConfig
from llama_index.core import ChatPromptTemplate

import pandas as pd
from llama_index.core import Document


# Khởi tạo Flask app
app = Flask(__name__)

# Quantization và configuration
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=False,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=False,
# )

# Load model, tokenizer và embedding
Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
    context_window=3900,
    max_new_tokens=256,
    # model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.3, "top_k": 50, "top_p": 0.95},
    device_map="cuda",
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)

# Load Excel file
import os
import pandas as pd

# Define the relative path from the script's location
file_path = os.path.join("retrieved_data", "thuyloi_ver1.xlsx")

# Check if the file exists and load it
if os.path.exists(file_path):
    print(f"File found at {file_path}. Attempting to load.")
    try:
        df = pd.read_excel(file_path)
        print("File loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
else:
    print(f"File not found at {file_path}")

    
# Load the Excel file
try:
    df = pd.read_excel(file_path)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"File not found at {file_path}")

# Load documents và create indexing
documents = [
    Document(
        text=row['data'], 
        metadata={"filename": "thuyloi_ver1", "category": row['source']}
    )
    for _, row in df.iterrows()
]  

# Replace with your actual documents
index = VectorStoreIndex.from_documents(documents)

# Chat prompt template
chat_text_qa_msgs = [
    (
        "user",
        """
        Bạn là nhà thông thái, biết rất nhiều về Đại học Thủy Lợi. Trường Đại học Thủy lợi là trường đại học số 1 trong việc đào tạo nguồn nhân lực chất lượng cao, nghiên cứu khoa học, phát triển và chuyển giao công nghệ tiên tiến trong các ngành khoa học, kỹ thuật, kinh tế và quản lý, đặc biệt trong lĩnh vực thủy lợi, môi trường, phòng chống và giảm nhẹ thiên tai.
        {context_str} 
        Question: 
        {query_str} """
    )
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# Initialize query engine
query_engine = index.as_query_engine(text_qa_template=text_qa_template)

# Define the Flask route for querying
@app.route('/get_text_response', methods=['POST'])
def query_model():
    # Get JSON request data
    data = request.get_json()
    query = data.get("query")

    # Handle empty query case
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Run the query on the engine
    try:
        response = query_engine.query(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
