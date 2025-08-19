#!/usr/bin/env python3
"""
Enhanced Paper Download Script
Handles HTML pages and extracts actual PDF links
"""

import json
import os
import requests
import time
import re
from urllib.parse import urlparse, urljoin
from pathlib import Path
import logging
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedPaperDownloader:
    def __init__(self, json_file='papers.json', download_dir='downloaded_papers_pdf'):
        self.json_file = json_file
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
        # Set headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def load_papers(self):
        """Load papers from JSON file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('papers', [])
        except Exception as e:
            logger.error(f"Error loading papers from {self.json_file}: {e}")
            return []
    
    def sanitize_filename(self, title, max_length=100):
        """Create a safe filename from paper title"""
        filename = re.sub(r'[^\w\s-]', '', title)
        filename = re.sub(r'[-\s]+', '_', filename)
        filename = filename.strip('_')
        
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename
    
    def extract_pdf_from_html(self, url, html_content):
        """Extract PDF link from HTML page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Common PDF link patterns
            pdf_patterns = [
                # Direct PDF links
                'a[href$=".pdf"]',
                'a[href*="/pdf/"]',
                'a[href*="download"]',
                # ArXiv specific
                'a[href*="arxiv.org/pdf"]',
                # PubMed specific
                'a[href*="pdf"]',
                # Generic download links
                'a[title*="PDF"]',
                'a[text*="PDF"]',
                'a.download-link',
                '.pdf-download',
                'a[href*="fulltext"]'
            ]
            
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            
            for pattern in pdf_patterns:
                links = soup.select(pattern)
                for link in links:
                    href = link.get('href')
                    if href:
                        # Make absolute URL
                        if href.startswith('http'):
                            pdf_url = href
                        else:
                            pdf_url = urljoin(url, href)
                        
                        logger.info(f"Found potential PDF link: {pdf_url}")
                        return pdf_url
            
            # For ArXiv pages, construct PDF URL
            if 'arxiv.org' in url:
                arxiv_match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
                    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # For PubMed, look for DOI and try to construct PDF URL
            if 'pubmed.ncbi.nlm.nih.gov' in url:
                doi_links = soup.find_all('a', href=re.compile(r'doi\.org'))
                if doi_links:
                    doi_url = doi_links[0]['href']
                    logger.info(f"Found DOI link: {doi_url}")
                    return doi_url  # Try the DOI link
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting PDF from HTML: {e}")
            return None
    
    def get_arxiv_pdf_url(self, arxiv_url):
        """Convert ArXiv page URL to PDF URL"""
        arxiv_patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)(v\d+)?',
            r'arxiv\.org/pdf/(\d+\.\d+)(v\d+)?',
            r'(\d+\.\d+)(v\d+)?'
        ]
        
        for pattern in arxiv_patterns:
            match = re.search(pattern, arxiv_url)
            if match:
                arxiv_id = match.group(1)
                version = match.group(2) or ''
                return f"https://arxiv.org/pdf/{arxiv_id}{version}.pdf"
        return None
    
    def download_file(self, url, filename, paper_title="Unknown"):
        """Download a file from URL with content type detection"""
        try:
            logger.info(f"Downloading: {paper_title}")
            logger.info(f"URL: {url}")
            
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if we got HTML instead of PDF
            if 'text/html' in content_type and 'pdf' not in url.lower():
                logger.warning(f"Got HTML instead of PDF for {url}")
                
                # Try to extract PDF link from HTML
                html_content = response.text
                pdf_url = self.extract_pdf_from_html(url, html_content)
                
                if pdf_url and pdf_url != url:
                    logger.info(f"Trying extracted PDF URL: {pdf_url}")
                    return self.download_file(pdf_url, filename, paper_title)
                else:
                    logger.warning(f"Could not find PDF link in HTML page: {url}")
                    return False
            
            # Determine file extension
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            elif 'html' in content_type:
                if not filename.endswith('.html'):
                    filename += '.html'
            
            file_path = self.download_dir / filename
            
            # Avoid overwriting existing files
            counter = 1
            original_path = file_path
            while file_path.exists():
                stem = original_path.stem
                suffix = original_path.suffix
                file_path = self.download_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Check if we actually got a PDF
            if file_path.suffix == '.pdf':
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        logger.warning(f"Downloaded file is not a valid PDF: {file_path}")
                        # Rename to .html if it's not a real PDF
                        html_path = file_path.with_suffix('.html')
                        file_path.rename(html_path)
                        file_path = html_path
            
            logger.info(f"Successfully downloaded: {file_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    def download_paper(self, paper):
        """Download a single paper with enhanced PDF detection"""
        title = paper.get('title', 'Unknown Title')
        safe_title = self.sanitize_filename(title)
        
        # Priority order for URL selection
        url_fields = [
            ('pdfUrl', 'Direct PDF URL'),
            ('arxivPageUrl', 'ArXiv URL'),
            ('url', 'General URL'),
            ('doi', 'DOI URL')
        ]
        
        for field, description in url_fields:
            url = paper.get(field)
            if url and url != 'N/A':
                logger.info(f"Trying {description}: {url}")
                
                # Special handling for ArXiv URLs
                if field == 'arxivPageUrl' and 'arxiv.org' in url:
                    pdf_url = self.get_arxiv_pdf_url(url)
                    if pdf_url:
                        if self.download_file(pdf_url, safe_title, title):
                            return True
                
                # Try the URL as-is
                if self.download_file(url, safe_title, title):
                    return True
        
        logger.warning(f"No downloadable URL found for: {title}")
        return False
    
    def download_all_papers(self):
        """Download all papers from the JSON file"""
        papers = self.load_papers()
        
        if not papers:
            logger.error("No papers found in the JSON file")
            return
        
        logger.info(f"Found {len(papers)} papers to download")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', f'Paper {i}')
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing paper {i}/{len(papers)}: {title[:50]}...")
            
            if self.download_paper(paper):
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Add delay between downloads to be respectful
            time.sleep(2)
        
        logger.info(f"\n" + "="*60)
        logger.info(f"Download Summary:")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Failed downloads: {failed_downloads}")
        logger.info(f"Total papers: {len(papers)}")
        logger.info(f"Files saved to: {self.download_dir.absolute()}")

def main():
    """Main function"""
    downloader = EnhancedPaperDownloader()
    
    print("Enhanced Paper Downloader")
    print("========================")
    print("This version tries to extract actual PDF links from HTML pages")
    print("\nStarting download process...")
    
    downloader.download_all_papers()

if __name__ == "__main__":
    main()
