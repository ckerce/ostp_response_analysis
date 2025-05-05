# ./extract_markdown_terms.py
import tarfile
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Download necessary NLTK resources (run once)
# Make sure we download ALL required resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class MarkdownAnalyzer:
    def __init__(self, tar_path, use_lemmatization=True):
        """
        Initialize with the path to the tar archive containing markdown files.
        
        Args:
            tar_path (str): Path to the tar archive
            use_lemmatization (bool): Whether to use lemmatization instead of stemming
        """
        self.tar_path = tar_path
        self.use_lemmatization = use_lemmatization
        
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stemmer = PorterStemmer()
            
        self.stop_words = set(stopwords.words('english'))
        self.document_terms = {}  # Dictionary to store term frequencies for each document
        self.corpus_terms = Counter()  # Counter for term frequencies across all documents
        
    def extract_and_analyze(self, max_files=None):
        """
        Extract and analyze markdown files in the tar archive.
        
        Args:
            max_files (int, optional): Maximum number of files to process.
                                      If None, process all files.
        """
        with tarfile.open(self.tar_path, 'r') as tar:
            md_files = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.md')]
            
            # Limit the number of files if specified
            if max_files is not None:
                md_files = md_files[:max_files]
                
            for member in md_files:
                # Extract the file content
                f = tar.extractfile(member)
                if f is not None:
                    print(f"Processing: {member.name}")
                    content = f.read().decode('utf-8')
                    self._analyze_document(member.name, content)
                        
    def _analyze_document(self, doc_name, content):
        """
        Analyze a single document for term frequencies.
        
        Args:
            doc_name (str): Document name/path
            content (str): Document content
        """
        # Clean and tokenize the text
        # Remove markdown formatting
        content = re.sub(r'#+ ', '', content)  # Remove headers
        content = re.sub(r'\[.*?\]\(.*?\)', '', content)  # Remove links
        content = re.sub(r'`.*?`', '', content)  # Remove inline code
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Remove code blocks
        
        # Tokenize
        # First split by whitespace for simple tokenization if NLTK tokenizer fails
        try:
            tokens = word_tokenize(content.lower())
        except Exception as e:
            print(f"Warning: NLTK tokenization failed ({str(e)}), using basic tokenization")
            # Fallback to basic tokenization
            tokens = re.findall(r'\b\w+\b', content.lower())
        
        # Remove punctuation and numbers
        tokens = [token for token in tokens if token.isalpha()]
        # Remove stopwords
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        # Process tokens (lemmatize or stem)
        if self.use_lemmatization:
            # Get POS tag for better lemmatization (if available)
            try:
                pos_tags = nltk.pos_tag(filtered_tokens)
                # Convert POS tags to WordNet format
                processed_terms = []
                for word, tag in pos_tags:
                    wn_tag = self._get_wordnet_pos(tag)
                    if wn_tag:
                        processed_terms.append(self.lemmatizer.lemmatize(word, wn_tag))
                    else:
                        processed_terms.append(self.lemmatizer.lemmatize(word))
            except Exception as e:
                print(f"Warning: POS tagging failed ({str(e)}), using basic lemmatization")
                processed_terms = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        else:
            # Use stemming
            processed_terms = [self.stemmer.stem(token) for token in filtered_tokens]
        
        # Count term frequencies for this document
        term_counts = Counter(processed_terms)
        self.document_terms[doc_name] = term_counts
        
        # Update corpus-wide term frequencies
        self.corpus_terms.update(processed_terms)
        
    def _get_wordnet_pos(self, treebank_tag):
        """
        Convert Penn Treebank POS tags to WordNet POS tags
        """
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return None
    
    def get_document_term_frequencies(self, doc_name=None):
        """
        Get term frequencies for a specific document or all documents.
        
        Args:
            doc_name (str, optional): Document name. If None, returns all documents.
            
        Returns:
            dict: Term frequencies for the specified document(s)
        """
        if doc_name:
            return self.document_terms.get(doc_name, {})
        return self.document_terms
    
    def get_corpus_term_frequencies(self, top_n=None):
        """
        Get term frequencies across all documents.
        
        Args:
            top_n (int, optional): If specified, returns only the top N terms.
            
        Returns:
            Counter: Term frequencies across all documents
        """
        if top_n:
            return Counter({term: count for term, count in self.corpus_terms.most_common(top_n)})
        return self.corpus_terms
    
    def generate_term_frequency_report(self, output_dir, top_n=20):
        """
        Generate a report of term frequencies.
        
        Args:
            output_dir (str): Directory to save the report
            top_n (int): Number of top terms to include
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create a DataFrame for corpus-wide term frequencies
        corpus_df = pd.DataFrame(self.corpus_terms.most_common(top_n), 
                                columns=['Term', 'Frequency'])
        
        # Save to CSV
        corpus_df.to_csv(output_path / 'corpus_term_frequencies.csv', index=False)
        
        # Try to create a plot if matplotlib is available with a proper backend
        try:
            # Use a non-GUI backend
            import matplotlib
            matplotlib.use('Agg')  # Set the Agg backend (non-GUI)
            
            plt.figure(figsize=(12, 8))
            plt.bar(corpus_df['Term'], corpus_df['Frequency'])
            plt.title('Top Terms Across All Documents')
            plt.xlabel('Term')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'top_terms.png')
            print(f"Plot saved to {output_path / 'top_terms.png'}")
        except Exception as e:
            print(f"Warning: Could not generate plot due to: {str(e)}")
            print("Continuing with CSV report generation only...")
        
        # Generate individual document reports
        document_terms_df = pd.DataFrame()
        
        for doc_name, term_counts in self.document_terms.items():
            doc_df = pd.DataFrame(term_counts.most_common(top_n),
                                 columns=['Term', f'Freq_{Path(doc_name).stem}'])
            if document_terms_df.empty:
                document_terms_df = doc_df
            else:
                document_terms_df = pd.merge(document_terms_df, doc_df, on='Term', how='outer')
        
        document_terms_df.fillna(0, inplace=True)
        document_terms_df.to_csv(output_path / 'document_term_frequencies.csv', index=False)
        
        return output_path


# Usage example
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract and analyze terms from markdown files in a tar archive')
    parser.add_argument('tar_path', help='Path to the tar archive containing markdown files')
    parser.add_argument('--max_files', type=int, default=2, help='Maximum number of files to process (default: 2, use 0 for all files)')
    parser.add_argument('--output_dir', default='./output', help='Directory to save the output report (default: ./output)')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top terms to include in the report (default: 20)')
    parser.add_argument('--use_stemming', action='store_true', help='Use stemming instead of lemmatization (default: use lemmatization)')
    
    args = parser.parse_args()
    
    # Adjust max_files if set to 0 (process all files)
    max_files = None if args.max_files == 0 else args.max_files
    
    print(f"Processing tar archive: {args.tar_path}")
    print(f"Max files to process: {'All' if max_files is None else max_files}")
    print(f"Using {'stemming' if args.use_stemming else 'lemmatization'} for word normalization")
    
    try:
        # Create analyzer and process the files
        analyzer = MarkdownAnalyzer(args.tar_path, use_lemmatization=not args.use_stemming)
        analyzer.extract_and_analyze(max_files=max_files)
        
        # Get overall term frequencies
        corpus_terms = analyzer.get_corpus_term_frequencies(top_n=args.top_n)
        print(f"\nTop {args.top_n} terms across all documents:")
        for term, count in corpus_terms.items():
            print(f"{term}: {count}")
        
        # Generate a report
        output_dir = analyzer.generate_term_frequency_report(args.output_dir, top_n=args.top_n)
        print(f"\nReport saved to {output_dir}")
        
        # Summary
        print(f"\nSummary:")
        print(f"Total documents processed: {len(analyzer.document_terms)}")
        print(f"Total unique terms: {len(analyzer.corpus_terms)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
