import os
import logging
from typing import List, Optional
import PyPDF2
from docx import Document
import pandas as pd
import re

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.supported_extensions = {
            '.pdf': self._extract_pdf_text,
            '.docx': self._extract_docx_text,
            '.txt': self._extract_txt_text,
            '.xlsx': self._extract_excel_text,
            '.csv': self._extract_csv_text
        }
    
    async def extract_text(self, file_path: str) -> str:
        """Extract text from file based on its extension"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            extractor = self.supported_extensions[file_ext]
            text = await extractor(file_path)
            
            # Clean and normalize text
            text = self._clean_text(text)
            
            logger.info(f"Successfully extracted text from {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            raise
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading TXT file with latin-1 encoding: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error extracting TXT text: {str(e)}")
            raise
    
    async def _extract_excel_text(self, file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text = ""
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += df.to_string(index=False) + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting Excel text: {str(e)}")
            raise
    
    async def _extract_csv_text(self, file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
            
        except Exception as e:
            logger.error(f"Error extracting CSV text: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove excessive newlines
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            # Remove special characters that might cause issues
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
            
            # Normalize spacing around punctuation
            text = re.sub(r'\s+([\.\,\!\?\;\:])', r'\1', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def get_file_info(self, file_path: str) -> dict:
        """Get information about a file"""
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found"}
            
            stat = os.stat(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            return {
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": stat.st_size,
                "file_extension": file_ext,
                "is_supported": file_ext in self.supported_extensions,
                "last_modified": stat.st_mtime
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {"error": str(e)} 