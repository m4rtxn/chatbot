"""This module loads HTML from files, cleans up, splits, and ingests into Weaviate."""

import logging
import os
import re
import weaviate
import glob
import csv

from chatbot_api.parsers import custom_site_extractor
from bs4 import BeautifulSoup, SoupStrainer
from chatbot_api.constants import WEAVIATE_DOCS_INDEX_NAME
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer  
from langchain.embeddings.base import Embeddings  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalHuggingFaceEmbeddings(Embeddings):  
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):  
        self.model = SentenceTransformer(model_name)  
      
    def embed_documents(self, texts):  
        return self.model.encode(texts).tolist()  
      
    def embed_query(self, text):  
        return self.model.encode([text])[0].tolist()  
  
def get_embeddings_model() -> Embeddings:  
    """Create and return a local Hugging Face embeddings model.  
      
    Returns:  
        Embeddings: The created embeddings model.  
    """  
    return LocalHuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    """Extract metadata from the given BeautifulSoup object.

    Args:
        meta (dict): The metadata to extract.
        soup (BeautifulSoup): The BeautifulSoup object to extract metadata from.

    Returns:
        dict: The extracted metadata.
    """
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_custom_site():
    """Load the custom site using a SitemapLoader.

    Returns:
        list: The loaded documents from the custom site.
    """
    return SitemapLoader(
        os.environ.get("LOAD_CUSTOM_SITE_URL"),
        filter_urls=[os.environ.get("FILTER_LOAD_CUSTOM_SITE_URL")],
        parsing_function=custom_site_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content", "description")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_custom_blog():
    """Load the custom blog using a RecursiveUrlLoader.

    Returns:
        list: The loaded documents from the custom blog.
    """
    return RecursiveUrlLoader(
        url=os.environ.get("LOAD_CUSTOM_BLOG_URL"),
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    """Extract text from the given HTML string.

    Args:
        html (str): The HTML string to extract text from.

    Returns:
        str: The extracted text.
    """
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_custom_stock():
    """Load all CSV files from the data directory with enhanced date handling and validation.

    Returns:
        list: The loaded documents from all CSV files in the data directory.
    """
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return []
    
    all_docs = []
    for csv_file in csv_files:
        logger.info(f"Loading CSV file: {csv_file}")
        try:
            # Default fieldnames for event data
            fieldnames = [
                "title",
                "building_name",
                "address",
                "month",
                "dates",
                "description",
                "category",
                "costs",
                "start_date",  # Added for better date handling
                "end_date",    # Added for better date handling
                "is_recurring", # Added for recurring events
                "recurrence_pattern", # Added for recurring events pattern
                "time_zone",   # Added for timezone support
            ]
            
            # Try to detect fieldnames from the CSV file
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    fieldnames = header

            def validate_and_transform_dates(row):
                """Validate and transform date-related fields in the row."""
                from datetime import datetime
                import pytz
                
                # Date formats to try
                date_formats = [
                    "%Y-%m-%d",
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                    "%Y/%m/%d",
                    "%d-%m-%Y",
                    "%m-%d-%Y",
                ]

                def parse_date(date_str):
                    if not date_str:
                        return None
                    
                    for fmt in date_formats:
                        try:
                            return datetime.strptime(date_str.strip(), fmt)
                        except ValueError:
                            continue
                    return None

                # Process dates field
                if row.get('dates'):
                    dates = row['dates'].split(',')
                    parsed_dates = []
                    for date in dates:
                        parsed_date = parse_date(date)
                        if parsed_date:
                            parsed_dates.append(parsed_date.strftime("%Y-%m-%d"))
                    row['dates'] = ','.join(parsed_dates) if parsed_dates else row['dates']

                # Process start_date and end_date
                for field in ['start_date', 'end_date']:
                    if row.get(field):
                        parsed_date = parse_date(row[field])
                        if parsed_date:
                            row[field] = parsed_date.strftime("%Y-%m-%d")

                # Validate timezone if present
                if row.get('time_zone'):
                    try:
                        pytz.timezone(row['time_zone'])
                    except pytz.exceptions.UnknownTimeZoneError:
                        row['time_zone'] = 'UTC'  # Default to UTC if invalid

                # Validate recurrence pattern if event is recurring
                if row.get('is_recurring') and row.get('is_recurring').lower() == 'true':
                    valid_patterns = ['daily', 'weekly', 'monthly', 'yearly']
                    if not row.get('recurrence_pattern') or row['recurrence_pattern'].lower() not in valid_patterns:
                        row['recurrence_pattern'] = 'none'

                return row

            # Custom CSV loader with data validation
            class ValidatedCSVLoader(CSVLoader):
                def load(self):
                    docs = super().load()
                    validated_docs = []
                    for doc in docs:
                        try:
                            # Convert the string content to a dictionary
                            content_dict = {}
                            for line in doc.page_content.split('\n'):
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    content_dict[key.strip()] = value.strip()
                            
                            # Apply validation and transformation
                            validated_content = validate_and_transform_dates(content_dict)
                            
                            # Convert back to string format
                            formatted_content = '\n'.join([f"{k}: {v}" for k, v in validated_content.items()])
                            doc.page_content = formatted_content
                            validated_docs.append(doc)
                        except Exception as e:
                            logger.error(f"Error processing document: {str(e)}")
                            # Keep the original document if validation fails
                            validated_docs.append(doc)
                    return validated_docs

            docs = ValidatedCSVLoader(
                file_path=csv_file,
                csv_args={
                    "delimiter": ",",
                    "fieldnames": fieldnames,
                },
            ).load()
            
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {csv_file}")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {str(e)}")
    
    return all_docs


def load_markdown_files():
    """Load all Markdown files from the data/markdown directory.

    Returns:
        list: The loaded documents from all Markdown files in the data/markdown directory.
    """
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    markdown_dir = os.path.join(data_dir, "markdown")
    markdown_files = glob.glob(os.path.join(markdown_dir, "*.md"))
    
    if not markdown_files:
        logger.warning(f"No Markdown files found in {markdown_dir}")
        return []
    
    all_docs = []
    for markdown_file in markdown_files:
        logger.info(f"Loading Markdown file: {markdown_file}")
        try:
            docs = UnstructuredMarkdownLoader(
                file_path=markdown_file,
            ).load()
            
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {markdown_file}")
        except Exception as e:
            logger.error(f"Error loading {markdown_file}: {str(e)}")
    
    return all_docs


def ingest_docs():
    """Ingest documents into Weaviate.

    This function loads documents from the custom site, custom stock, custom blog, and markdown files,
    transforms them, and ingests them into Weaviate.
    """
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    #docs_from_custom = load_custom_site()
    #logger.info(f"Loaded {len(docs_from_custom)} docs from custom")
    docs_from_custom_stock = load_custom_stock()
    logger.info(f"Loaded {len(docs_from_custom_stock)} docs from custom stock")
    #docs_from_custom_blog = load_custom_blog()
    #logger.info(f"Loaded {len(docs_from_custom_blog)} docs from custom blog")
    docs_from_markdown = load_markdown_files()
    logger.info(f"Loaded {len(docs_from_markdown)} docs from markdown files")

    #docs_transformed = text_splitter.split_documents(
    #    docs_from_custom + docs_from_custom_stock + docs_from_custom_blog
    #)
    docs_transformed = text_splitter.split_documents(
        docs_from_custom_stock + docs_from_markdown
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )


if __name__ == "__main__":
    """Main execution point of the application.

    This block is executed when the module is run directly, not when it is imported.
    It starts the ingestion of documents into Weaviate.
    """
    ingest_docs()
