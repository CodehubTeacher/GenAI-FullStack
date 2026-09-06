import base64
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import pandas as pd
import pdfplumber
import io
from typing import List
from pathlib import Path
import faiss
import numpy as np

class StructuredDataLoader:

    # Load data from structured sources like csv, json etc. 

    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            **kwargs: Additional arguments for pandas.read_csv.
        
        Returns:
            pd.DataFrame: Loaded data.
        """
        return pd.read_csv(file_path, **kwargs)

    def load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            **kwargs: Additional arguments for pandas.read_json.
        
        Returns:
            pd.DataFrame: Loaded data.
        """
        return pd.read_json(file_path, **kwargs)

    def load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to the Excel file.
            **kwargs: Additional arguments for pandas.read_excel.
        
        Returns:
            pd.DataFrame: Loaded data.
        """
        return pd.read_excel(file_path, **kwargs)


class UnstructuredDataLoader:
    def __init__(self, max_chunk_size: int = 3000):
        """
        Initialize the generic text and image loader/chunker.
        
        Args:
            max_chunk_size: Maximum size of each text chunk (in characters)
        """
        self.max_chunk_size = max_chunk_size

    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, List[str]]:
        """
        Extract text and images from PDF document.
        
        Returns:
            tuple: (extracted_text, list_of_base64_images)
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    img_bytes = pix.tobytes("png")
                    img_pil = Image.open(io.BytesIO(img_bytes))
                    img_array = img_pil.convert("RGB")
                    pixels = list(img_array.getdata())
                    if len(set(pixels)) > 1:
                        img_data = img_bytes
                    else:
                        continue
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        images_folder = "extracted_images"
                        Path(images_folder).mkdir(exist_ok=True)
                        img_filename = f"page_{page_num+1}_img_{img_index+1}.png"
                        img_path = Path(images_folder) / img_filename
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        base64_img = base64.b64encode(img_data).decode()
                        images.append(base64_img)
                    pix = None
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num}: {e}")
        doc.close()
        return full_text, images
    
    def chunk_text(self, text: str, overlap: int = 200) -> List[str]:
        """
        Split text into chunks for processing, with optional overlap between chunks.
        
        Args:
            text: The input text to chunk
            overlap: Number of characters to overlap between consecutive chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self.max_chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == text_length:
                break
            start = end - overlap  # move start back by overlap
        return chunks

    def extract_tables_from_pdf(self, pdf_path: str):
        """Extract tables from PDF and return a tuple: (Markdown table chunks, list of pandas DataFrames)."""
        table_chunks = []
        tables_df = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    # Convert table to Markdown format
                    md_rows = []
                    if table and len(table) > 0:
                        header = ["**" + (str(cell) if cell is not None else "") + "**" for cell in table[0]]
                        md_rows.append("| " + " | ".join(header) + " |")
                        md_rows.append("|" + " --- |" * len(header))
                        for row in table[1:]:
                            md_row = [str(cell) if cell is not None else "" for cell in row]
                            md_rows.append("| " + " | ".join(md_row) + " |")
                    md_str = f"--- Table {table_idx+1} on Page {page_num+1} ---\n" + "\n".join(md_rows)
                    table_chunks.append(md_str)
                    # Add DataFrame
                    if table and len(table) > 0:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables_df.append(df)
        return table_chunks, tables_df

class FaissRetriever:
    """
    Simple FAISS-based retriever for text embeddings.
    Stores embeddings and retrieves top-k similar items.
    """

    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Dimension of the embedding vectors.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []

    def add(self, embeddings: np.ndarray, texts: List[str]):
        """
        Add embeddings and corresponding texts to the index.

        Args:
            embeddings: np.ndarray of shape (n_samples, embedding_dim)
            texts: List of strings corresponding to embeddings
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch.")
        self.index.add(embeddings)
        self.texts.extend(texts)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k most similar texts for the query embedding.

        Args:
            query_embedding: np.ndarray of shape (1, embedding_dim)
            top_k: Number of results to return

        Returns:
            List of top-k similar texts
        """
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError("Query embedding dimension mismatch.")
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]
