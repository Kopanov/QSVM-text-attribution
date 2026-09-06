import os
import pandas as pd
import numpy as np
import re
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from typing import Dict, Tuple, Any, Union, List, Optional
from scipy.sparse import csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import string
import io

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data loading, preprocessing, and feature extraction for QSVM project
    """
    
    def __init__(self):
        """Initialize the data processor with specified parameters"""
        # TF-IDF parameters as specified in the requirements
        self.tfidf_params = {
            'max_features': 3000,
            'ngram_range': (1, 2),
            'min_df': 5,
            'max_df': 0.7,
            'stop_words': 'english',
            'lowercase': True,
            'strip_accents': 'ascii'
        }
        
        self.vectorizer = None
        self.label_encoder = None

        # Multi-feature extraction enabled
        self.use_multi_features = True
        self.feature_scaler = StandardScaler()

        # Feature mode (controlled via FEATURE_MODE env var):
        #   'current'            - baseline: word TF-IDF + 18 linguistic → PCA inside quantum_svm
        #   'hybrid'             - char 3-5gram TF-IDF → internal PCA(8) + top-6 MI stylometric → 14 dims, skip_pca
        #   'stylometric_direct' - 24 extended features → MI selection → top-N dims, skip_pca
        self.feature_mode = os.environ.get('FEATURE_MODE', 'current')
        self.char_vectorizer = None   # hybrid mode: fitted char n-gram TF-IDF
        self.char_pca = None          # hybrid mode: internal PCA on char features
        self.mi_selector = None       # hybrid / direct mode: fitted MI feature selector
        self.mi_selected_feature_names = None  # names of MI-selected features

        # Hedge / discourse markers for hedge_ratio feature
        self._hedge_words = [
            "however", "nevertheless", "nonetheless", "although", "whereas",
            "it's important to note", "it is important to note",
            "it's worth noting", "it is worth noting",
            "keep in mind", "please note", "importantly",
            "i believe", "i think", "in my opinion", "in my view",
            "it's possible", "it is possible", "this may vary",
            "generally speaking", "broadly speaking", "in general",
            "typically", "usually", "often", "sometimes",
        ]

        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception:
            logger.warning("Could not download NLTK data, using basic preprocessing")
    
    def load_csv_with_flexible_delimiter(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file with automatic delimiter detection and UTF-8 encoding
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pandas DataFrame
        """
        try:
            # Try semicolon delimiter first (common in European CSV files)
            try:
                df_semicolon = pd.read_csv(file_path, sep=';', encoding='utf-8-sig', encoding_errors='replace')
                # Check if we have the expected columns
                if len(df_semicolon.columns) >= 4 and 'category' in df_semicolon.columns:
                    logger.info("Successfully loaded CSV with semicolon delimiter")
                    return df_semicolon
            except Exception as e:
                logger.warning(f"Semicolon delimiter failed: {e}")
            
            # Try comma delimiter as fallback
            try:
                df_comma = pd.read_csv(file_path, sep=',', encoding='utf-8', encoding_errors='replace')
                # Check if we have the expected columns
                if len(df_comma.columns) >= 4:
                    logger.info("Successfully loaded CSV with comma delimiter")
                    return df_comma
            except Exception as e:
                logger.warning(f"Comma delimiter failed: {e}")
            
            # Try UTF-8 without BOM as last resort
            try:
                df_utf8 = pd.read_csv(file_path, sep=';', encoding='utf-8', encoding_errors='replace')
                if len(df_utf8.columns) >= 4:
                    logger.info("Successfully loaded CSV with UTF-8 encoding")
                    return df_utf8
            except Exception as e:
                logger.error(f"All CSV loading attempts failed: {e}")
                # Return empty DataFrame as fallback
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short responses (likely incomplete)
        if len(text.split()) < 3:
            return ""
        
        return text
    
    def extract_features_labels(self, df: pd.DataFrame) -> Tuple[list, list, list]:
        """
        Extract features (text responses), labels, and categories from DataFrame
        
        Args:
            df: Input DataFrame with columns 'category', 'Qwen2.5-32B' and 'Gemma3-27B'
            
        Returns:
            Tuple of (texts, labels, categories) where texts are combined responses, 
            labels indicate model, and categories track the writing category
        """
        texts = []
        labels = []
        categories = []
        
        # Process each row to maintain category information
        for _, row in df.iterrows():
            category = row.get('category', 'Unknown')
            qwen_text = self.clean_text(str(row.get('Qwen2.5-32B', '')))
            gemma_text = self.clean_text(str(row.get('Gemma3-27B', '')))
            
            if qwen_text:  # Only include non-empty responses
                texts.append(qwen_text)
                labels.append('Qwen')
                categories.append(category)
            
            if gemma_text:  # Only include non-empty responses
                texts.append(gemma_text)
                labels.append('Gemma')
                categories.append(category)
        
        logger.info(f"Extracted {len(texts)} text samples: {labels.count('Qwen')} Qwen, {labels.count('Gemma')} Gemma")
        
        # Log category distribution
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        logger.info(f"Category distribution: {category_counts}")
        
        return texts, labels, categories
    
    def extract_lexical_features(self, text: str) -> Dict[str, float]:
        """Extract lexical (word-level) features"""
        if not text or len(text.strip()) == 0:
            return {f'lexical_{k}': 0.0 for k in ['avg_word_len', 'avg_sent_len', 'unique_ratio', 
                                                  'vocab_richness', 'word_count', 'char_count']}
        
        # Basic text statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Filter out punctuation-only tokens
        content_words = [w for w in words if w.isalnum()]
        
        if len(content_words) == 0:
            return {f'lexical_{k}': 0.0 for k in ['avg_word_len', 'avg_sent_len', 'unique_ratio', 
                                                  'vocab_richness', 'word_count', 'char_count']}
        
        features = {
            'lexical_avg_word_len': np.mean([len(word) for word in content_words]),
            'lexical_avg_sent_len': len(content_words) / max(1, len(sentences)),
            'lexical_unique_ratio': len(set(content_words)) / len(content_words),
            'lexical_vocab_richness': len(set(content_words)) / np.sqrt(len(content_words)),
            'lexical_word_count': len(content_words),
            'lexical_char_count': len(text)
        }
        
        return features
    
    def extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """Extract syntactic (grammar-level) features"""
        if not text or len(text.strip()) == 0:
            return {f'syntactic_{k}': 0.0 for k in ['noun_ratio', 'verb_ratio', 'adj_ratio', 
                                                   'adv_ratio', 'punct_density', 'complexity']}
        
        try:
            words = word_tokenize(text)
            # Simple POS tagging fallback if full NLTK not available
            pos_tags = nltk.pos_tag(words) if len(words) > 0 else []
            
            if len(pos_tags) == 0:
                return {f'syntactic_{k}': 0.0 for k in ['noun_ratio', 'verb_ratio', 'adj_ratio', 
                                                       'adv_ratio', 'punct_density', 'complexity']}
                
            # POS tag categories
            pos_counts = Counter([tag for word, tag in pos_tags])
            total_tags = len(pos_tags)
            
            # Calculate ratios
            noun_ratio = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                         pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_tags
            
            verb_ratio = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                         pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                         pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_tags
            
            adj_ratio = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                        pos_counts.get('JJS', 0)) / total_tags
            
            adv_ratio = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                        pos_counts.get('RBS', 0)) / total_tags
            
            # Punctuation density
            punct_count = sum(1 for char in text if char in string.punctuation)
            punct_density = punct_count / max(1, len(text))
            
            # Syntactic complexity (rough measure)
            sentences = sent_tokenize(text)
            avg_clause_length = len(words) / max(1, len(sentences))
            complexity_score = avg_clause_length / 20.0  # Normalize
            
            features = {
                'syntactic_noun_ratio': noun_ratio,
                'syntactic_verb_ratio': verb_ratio,
                'syntactic_adj_ratio': adj_ratio,
                'syntactic_adv_ratio': adv_ratio,
                'syntactic_punct_density': punct_density,
                'syntactic_complexity': min(complexity_score, 1.0)  # Cap at 1.0
            }
            
        except Exception as e:
            logger.warning(f"Syntactic feature extraction failed: {e}")
            features = {f'syntactic_{k}': 0.0 for k in ['noun_ratio', 'verb_ratio', 'adj_ratio', 
                                                       'adv_ratio', 'punct_density', 'complexity']}
        
        return features
    
    def extract_stylistic_features(self, text: str) -> Dict[str, float]:
        """Extract stylistic (writing style) features - KEY for Qwen vs Gemma differentiation"""
        if not text or len(text.strip()) == 0:
            return {f'stylistic_{k}': 0.0 for k in ['formality', 'question_ratio', 'exclaim_ratio',
                                                   'capital_ratio', 'contraction_ratio', 'list_markers']}
        
        # Formality indicators
        formal_words = ['therefore', 'furthermore', 'however', 'consequently', 'moreover', 
                       'nevertheless', 'subsequently', 'accordingly', 'specifically']
        informal_words = ['yeah', 'okay', 'cool', 'awesome', 'totally', 'basically', 
                         'like', 'you know', 'I mean', 'sort of']
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_words if word in text_lower)
        informal_count = sum(1 for word in informal_words if word in text_lower)
        
        # Calculate formality score
        total_indicators = formal_count + informal_count
        if total_indicators > 0:
            formality_score = formal_count / total_indicators
        else:
            # Fallback: check sentence structure complexity
            sentences = sent_tokenize(text)
            avg_sent_len = np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0
            formality_score = min(avg_sent_len / 25.0, 1.0)  # Longer sentences = more formal
        
        # Punctuation patterns
        question_ratio = text.count('?') / max(1, len(sent_tokenize(text)))
        exclaim_ratio = text.count('!') / max(1, len(sent_tokenize(text)))
        
        # Capitalization patterns
        if len(text) > 0:
            capital_ratio = sum(1 for c in text if c.isupper()) / len(text)
        else:
            capital_ratio = 0.0
        
        # Contractions (informal style marker)
        contractions = ["n't", "'re", "'ve", "'ll", "'d", "'s", "'m"]
        contraction_count = sum(text.lower().count(contr) for contr in contractions)
        contraction_ratio = contraction_count / max(1, len(word_tokenize(text)))
        
        # List markers (structural style)
        list_markers = re.findall(r'^\s*[-•*]\s', text, re.MULTILINE)
        numbered_lists = re.findall(r'^\s*\d+[.)]\s', text, re.MULTILINE)
        list_marker_density = (len(list_markers) + len(numbered_lists)) / max(1, len(sent_tokenize(text)))
        
        features = {
            'stylistic_formality': formality_score,
            'stylistic_question_ratio': min(question_ratio, 1.0),
            'stylistic_exclaim_ratio': min(exclaim_ratio, 1.0),
            'stylistic_capital_ratio': capital_ratio,
            'stylistic_contraction_ratio': min(contraction_ratio, 1.0),
            'stylistic_list_markers': min(list_marker_density, 1.0)
        }

        return features

    def extract_extended_stylistic_features(self, text: str) -> Dict[str, float]:
        """6 additional stylometric features for expanded direct encoding (Test B).

        These complement the existing 18 features to form the 24-feature set used
        in 'stylometric_direct' mode before mutual-information selection.
        """
        if not text or len(text.strip()) == 0:
            return {k: 0.0 for k in [
                'ext_digit_ratio', 'ext_sent_len_std', 'ext_hedge_ratio',
                'ext_paragraph_ratio', 'ext_repeated_bigram_ratio', 'ext_word_len_std'
            ]}

        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        content_words = [w for w in words if w.isalnum()]
        text_lower = text.lower()

        # 1. digit_ratio: proportion of digit characters in text
        digit_ratio = sum(c.isdigit() for c in text) / max(1, len(text))

        # 2. sent_len_std: std dev of sentence lengths (tokens) — captures length variation
        sent_lens = [len(word_tokenize(s)) for s in sentences] if sentences else [0]
        sent_len_std = float(np.std(sent_lens)) / 20.0  # normalize by typical std scale

        # 3. hedge_ratio: count of hedging/disclaimer phrases per sentence
        hedge_count = sum(1 for phrase in self._hedge_words if phrase in text_lower)
        hedge_ratio = hedge_count / max(1, len(sentences))

        # 4. paragraph_ratio: number of distinct paragraphs relative to sentence count
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_ratio = len(paragraphs) / max(1, len(sentences))

        # 5. repeated_bigram_ratio: fraction of bigrams that are repeats
        if len(content_words) >= 2:
            bigrams = list(zip(content_words[:-1], content_words[1:]))
            bigram_counts = Counter(bigrams)
            repeated = sum(cnt - 1 for cnt in bigram_counts.values() if cnt > 1)
            repeated_bigram_ratio = repeated / max(1, len(bigrams))
        else:
            repeated_bigram_ratio = 0.0

        # 6. word_len_std: std dev of content word lengths — lexical variety
        word_len_std = float(np.std([len(w) for w in content_words])) / 5.0 if content_words else 0.0

        return {
            'ext_digit_ratio':            min(digit_ratio, 1.0),
            'ext_sent_len_std':           min(sent_len_std, 1.0),
            'ext_hedge_ratio':            min(hedge_ratio, 1.0),
            'ext_paragraph_ratio':        min(paragraph_ratio, 1.0),
            'ext_repeated_bigram_ratio':  min(repeated_bigram_ratio, 1.0),
            'ext_word_len_std':           min(word_len_std, 1.0),
        }

    def extract_extended_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract 24 features: 18 existing linguistic + 6 extended stylometric."""
        logger.info(f"Extracting extended linguistic features (24) from {len(texts)} texts...")
        rows = []
        for i, text in enumerate(texts):
            if i % 500 == 0:
                logger.info(f"  Extended features {i+1}/{len(texts)}...")
            base = self.extract_lexical_features(text)
            syn  = self.extract_syntactic_features(text)
            sty  = self.extract_stylistic_features(text)
            ext  = self.extract_extended_stylistic_features(text)
            rows.append(list({**base, **syn, **sty, **ext}.values()))
        arr = np.array(rows)
        logger.info(f"Extended feature extraction complete: shape={arr.shape}")
        return arr

    def get_extended_linguistic_feature_names(self) -> List[str]:
        """Names for all 24 features (18 base + 6 extended)."""
        return self.get_linguistic_feature_names() + [
            'ext_digit_ratio', 'ext_sent_len_std', 'ext_hedge_ratio',
            'ext_paragraph_ratio', 'ext_repeated_bigram_ratio', 'ext_word_len_std',
        ]

    def extract_multi_features(self, texts: List[str]) -> np.ndarray:
        """Extract multi-feature representation combining TF-IDF + linguistic features"""
        try:
            # 1. TF-IDF features
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(**self.tfidf_params)
                tfidf_features = self.vectorizer.fit_transform(texts)
            else:
                tfidf_features = self.vectorizer.transform(texts)
            
            # Convert sparse to dense
            from scipy.sparse import issparse
            if issparse(tfidf_features):
                tfidf_dense = tfidf_features.toarray()
            else:
                tfidf_dense = np.asarray(tfidf_features)
            
            # 2. Linguistic features if enabled
            if self.use_multi_features:
                linguistic_features = self.extract_linguistic_features(texts)
                from sklearn.exceptions import NotFittedError
                try:
                    linguistic_scaled = self.feature_scaler.transform(linguistic_features)
                except NotFittedError:
                    linguistic_scaled = self.feature_scaler.fit_transform(linguistic_features)
                    logger.info("StandardScaler fitted on current batch (first call)")
                
                # Combine features
                combined_features = np.hstack([tfidf_dense, linguistic_scaled])
                logger.info(f"Combined features: {tfidf_dense.shape[1]} TF-IDF + {linguistic_scaled.shape[1]} linguistic = {combined_features.shape[1]} total")
                return combined_features
            else:
                return tfidf_dense
                
        except Exception as e:
            logger.error(f"Multi-feature extraction failed: {e}")
            # Fallback to basic TF-IDF
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(**self.tfidf_params)
                return self.vectorizer.fit_transform(texts).toarray()
            else:
                return self.vectorizer.transform(texts).toarray()
    
    def process_data(self, df: pd.DataFrame, sample_size: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main method for data processing - creates prepare_data compatibility"""
        return self.prepare_data(df, sample_size)
    
    def prepare_data(self, df: pd.DataFrame, sample_size: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with multi-feature extraction.

        Splits raw texts BEFORE fitting TF-IDF / scaler to avoid data leakage.
        """
        try:
            logger.info(f"Processing data with sample_size: {sample_size}")

            # Extract text and labels
            X_raw, y, _ = self.extract_features_labels(df)

            # Apply sampling if specified
            if sample_size and sample_size != 'full':
                try:
                    sample_num = int(sample_size)
                    if sample_num < len(X_raw):
                        from sklearn.model_selection import train_test_split
                        X_raw, _, y, _ = train_test_split(X_raw, y,
                                                        train_size=sample_num,
                                                        stratify=y,
                                                        random_state=42)
                        logger.info(f"Applied stratified sampling: {len(X_raw)} samples")
                except Exception as e:
                    logger.warning(f"Sampling failed: {e}, using full dataset")

            # Split raw texts FIRST so TF-IDF/scaler see only training data
            from sklearn.model_selection import train_test_split
            X_raw_train, X_raw_test, y_train, y_test = train_test_split(
                X_raw, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # Fit on training, transform both (extract_multi_features fits on first call)
            X_train = self.extract_multi_features(X_raw_train)
            X_test = self.extract_multi_features(X_raw_test)

            logger.info(f"Data preparation complete: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract all linguistic features (lexical + syntactic + stylistic)"""
        logger.info(f"Extracting linguistic features from {len(texts)} texts...")
        
        linguistic_features = []
        for i, text in enumerate(texts):
            if i % 500 == 0:
                logger.info(f"Processing linguistic features {i+1}/{len(texts)}...")
                
            lex_feat = self.extract_lexical_features(text)
            syn_feat = self.extract_syntactic_features(text)
            sty_feat = self.extract_stylistic_features(text)
            
            # Combine all linguistic features
            combined_feat = {**lex_feat, **syn_feat, **sty_feat}
            linguistic_features.append(list(combined_feat.values()))
        
        linguistic_array = np.array(linguistic_features)
        logger.info(f"Extracted {linguistic_array.shape[1]} linguistic features")
        
        return linguistic_array
    
    def get_linguistic_feature_names(self) -> List[str]:
        """Get names of all linguistic features"""
        lexical_names = ['lexical_avg_word_len', 'lexical_avg_sent_len', 'lexical_unique_ratio', 
                        'lexical_vocab_richness', 'lexical_word_count', 'lexical_char_count']
        syntactic_names = ['syntactic_noun_ratio', 'syntactic_verb_ratio', 'syntactic_adj_ratio', 
                          'syntactic_adv_ratio', 'syntactic_punct_density', 'syntactic_complexity']
        stylistic_names = ['stylistic_formality', 'stylistic_question_ratio', 'stylistic_exclaim_ratio',
                          'stylistic_capital_ratio', 'stylistic_contraction_ratio', 'stylistic_list_markers']
        
        return lexical_names + syntactic_names + stylistic_names
    
    def load_and_preprocess(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Load and preprocess the dataset with NO DATA LEAKAGE - split prompts first
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing processed data splits and metadata
        """
        logger.info("Starting data preprocessing...")
        
        # CRITICAL FIX: Split prompts first to prevent data leakage
        # Split the original DataFrame (prompts) before creating text pairs
        
        prompt_categories = df['category'].values
        
        # Split prompts into train/validation first
        train_prompts, val_prompts = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=prompt_categories
        )
        
        logger.info(f"Train-validation split: {len(train_prompts)} train prompts, {len(val_prompts)} validation prompts")
        
        # Extract text pairs from each split separately (NO OVERLAP POSSIBLE)
        train_texts, train_labels, train_cats = self.extract_features_labels(train_prompts)
        val_texts, val_labels, val_cats = self.extract_features_labels(val_prompts)
        
        if len(train_texts) == 0 or len(val_texts) == 0:
            raise ValueError("No valid text samples found in training or validation split")
        
        # Log distributions to verify no leakage
        from collections import Counter
        train_cat_dist = Counter(train_cats)
        val_cat_dist = Counter(val_cats)
        
        logger.info(f"Training categories: {dict(train_cat_dist)}")
        logger.info(f"Validation categories: {dict(val_cat_dist)}")
        
        # Encode labels consistently
        self.label_encoder = LabelEncoder()
        # Fit on combined labels to ensure consistent encoding
        all_labels = train_labels + val_labels
        self.label_encoder.fit(all_labels)
        
        y_train = self.label_encoder.transform(train_labels)
        y_val = self.label_encoder.transform(val_labels)
        
        # Convert to arrays
        X_train_text = np.array(train_texts)
        X_val_text = np.array(val_texts)
        train_cats = np.array(train_cats) 
        val_cats = np.array(val_cats)
        
        logger.info(f"Train-validation split: {len(X_train_text)} train, {len(X_val_text)} validation")
        
        # Log stratification success
        train_cat_dist = {}
        val_cat_dist = {}
        for cat in train_cats:
            train_cat_dist[cat] = train_cat_dist.get(cat, 0) + 1
        for cat in val_cats:
            val_cat_dist[cat] = val_cat_dist.get(cat, 0) + 1
            
        logger.info(f"Training categories: {train_cat_dist}")
        logger.info(f"Validation categories: {val_cat_dist}")
        
        # 1. Fit TF-IDF vectorizer on training data only
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        X_val_tfidf = self.vectorizer.transform(X_val_text)
        
        # Convert to dense arrays
        from scipy.sparse import issparse
        if issparse(X_train_tfidf):
            X_train_tfidf_dense = X_train_tfidf.toarray()
        else:
            X_train_tfidf_dense = np.asarray(X_train_tfidf)
        
        if issparse(X_val_tfidf):
            X_val_tfidf_dense = X_val_tfidf.toarray()
        else:
            X_val_tfidf_dense = np.asarray(X_val_tfidf)
        
        # 2. Extract linguistic features if enabled
        if self.use_multi_features:
            logger.info("Multi-feature mode enabled - extracting linguistic features...")
            
            # Extract linguistic features
            train_linguistic = self.extract_linguistic_features(X_train_text.tolist())
            val_linguistic = self.extract_linguistic_features(X_val_text.tolist())
            
            # Scale linguistic features
            train_linguistic_scaled = self.feature_scaler.fit_transform(train_linguistic)
            val_linguistic_scaled = self.feature_scaler.transform(val_linguistic)
            
            # Combine TF-IDF + linguistic features
            X_train_dense = np.hstack([X_train_tfidf_dense, train_linguistic_scaled])
            X_val_dense = np.hstack([X_val_tfidf_dense, val_linguistic_scaled])
            
            logger.info(f"Multi-feature extraction completed: {X_train_tfidf_dense.shape[1]} TF-IDF + {train_linguistic_scaled.shape[1]} linguistic = {X_train_dense.shape[1]} total features")
        else:
            # TF-IDF only mode
            X_train_dense = X_train_tfidf_dense
            X_val_dense = X_val_tfidf_dense
            logger.info(f"TF-IDF-only extraction completed: {X_train_dense.shape[1]} features")
        
        # Calculate class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(self.label_encoder.inverse_transform(unique), counts))

        # Prepare feature names (baseline)
        tfidf_feature_names = self.vectorizer.get_feature_names_out().tolist()
        if self.use_multi_features:
            linguistic_feature_names = self.get_linguistic_feature_names()
            all_feature_names = tfidf_feature_names + linguistic_feature_names
        else:
            all_feature_names = tfidf_feature_names

        # -----------------------------------------------------------------------
        # FEATURE MODE BRANCHES
        # 'current'            → baseline: return combined TF-IDF + linguistic
        # 'hybrid'             → char 3-5gram TF-IDF PCA(8) + top-6 MI stylometric
        # 'stylometric_direct' → 24 extended features → top-N MI selection, no PCA
        # -----------------------------------------------------------------------
        skip_pca = False

        if self.feature_mode == 'hybrid':
            total_dim = int(os.environ.get('QUANTUM_FEATURE_DIM', 14))
            # Split: ~57% char-PCA (matches original 8/14 ratio), rest goes to MI stylometric
            char_k = max(2, round(total_dim * 8 / 14))
            mi_k   = total_dim - char_k
            logger.info(f"FEATURE_MODE=hybrid: char 3-5gram TF-IDF PCA({char_k}) + top-{mi_k} MI stylometric → {total_dim} dims")

            # Char n-gram TF-IDF (300 features)
            self.char_vectorizer = TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 5), max_features=300,
                min_df=2, max_df=0.95, lowercase=True
            )
            X_train_char = self.char_vectorizer.fit_transform(X_train_text.tolist()).toarray()
            X_val_char   = self.char_vectorizer.transform(X_val_text.tolist()).toarray()

            # Internal PCA on char features — dimension driven by QUANTUM_FEATURE_DIM
            self.char_pca = PCA(n_components=char_k, random_state=42)
            X_train_char_pca = self.char_pca.fit_transform(X_train_char)
            X_val_char_pca   = self.char_pca.transform(X_val_char)
            logger.info(f"Char TF-IDF PCA({char_k}) variance explained: {self.char_pca.explained_variance_ratio_.sum():.3f}")

            # 24 extended stylometric features → MI select top mi_k
            train_ext = self.extract_extended_linguistic_features(X_train_text.tolist())
            val_ext   = self.extract_extended_linguistic_features(X_val_text.tolist())
            ext_names = self.get_extended_linguistic_feature_names()
            self.mi_selector = SelectKBest(mutual_info_classif, k=mi_k)
            X_train_ling_sel = self.mi_selector.fit_transform(train_ext, y_train)
            X_val_ling_sel   = self.mi_selector.transform(val_ext)
            sel_indices = self.mi_selector.get_support(indices=True)
            self.mi_selected_feature_names = [ext_names[i] for i in sel_indices]
            logger.info(f"Hybrid MI-selected top-{mi_k} stylometric: {self.mi_selected_feature_names}")

            # Final combined: char_k char-PCA dims + mi_k direct stylometric = total_dim
            X_train_dense = np.hstack([X_train_char_pca, X_train_ling_sel])
            X_val_dense   = np.hstack([X_val_char_pca,   X_val_ling_sel])
            all_feature_names = (
                [f'char_pca_{i}' for i in range(char_k)] + self.mi_selected_feature_names
            )
            skip_pca = True
            logger.info(f"Hybrid feature matrix: {X_train_dense.shape[1]} dims ({char_k} char-PCA + {mi_k} stylometric), skip_pca=True")

        elif self.feature_mode == 'stylometric_direct':
            n_select = int(os.environ.get('QUANTUM_FEATURE_DIM', 14))
            logger.info(f"FEATURE_MODE=stylometric_direct: 24 extended features → MI top-{n_select}, skip_pca=True")

            train_ext = self.extract_extended_linguistic_features(X_train_text.tolist())
            val_ext   = self.extract_extended_linguistic_features(X_val_text.tolist())
            ext_names = self.get_extended_linguistic_feature_names()

            self.mi_selector = SelectKBest(mutual_info_classif, k=n_select)
            X_train_dense = self.mi_selector.fit_transform(train_ext, y_train)
            X_val_dense   = self.mi_selector.transform(val_ext)
            sel_indices = self.mi_selector.get_support(indices=True)
            self.mi_selected_feature_names = [ext_names[i] for i in sel_indices]
            # Sort by MI score descending for logging
            mi_scores = self.mi_selector.scores_
            ranked = sorted(zip(self.mi_selected_feature_names, mi_scores[sel_indices]),
                            key=lambda x: -x[1])
            logger.info(f"Direct MI-selected top-{n_select} features (by MI score):")
            for fname, score in ranked:
                logger.info(f"  {fname:<40} MI={score:.4f}")

            all_feature_names = self.mi_selected_feature_names
            skip_pca = True

        return {
            'X_train': X_train_dense,
            'X_val': X_val_dense,
            'y_train': y_train,
            'y_val': y_val,
            'X_train_text': X_train_text,
            'X_val_text': X_val_text,
            'train_categories': train_cats,
            'val_categories': val_cats,
            'feature_names': np.array(all_feature_names),
            'label_encoder': self.label_encoder,
            'class_distribution': np.array(list(class_distribution.values())),
            'class_names': list(class_distribution.keys()),
            'vectorizer': self.vectorizer,
            'use_multi_features': self.use_multi_features,
            'n_tfidf_features': len(tfidf_feature_names),
            'n_linguistic_features': len(linguistic_feature_names) if self.use_multi_features else 0,
            'skip_pca': skip_pca,
            'feature_mode': self.feature_mode,
            'mi_selected_feature_names': self.mi_selected_feature_names,
        }
    
    def get_feature_importance(self, model, top_k: int = 20) -> Dict[str, float]:
        """
        Get top feature importance from trained model
        
        Args:
            model: Trained model with feature importance
            top_k: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not hasattr(model, 'coef_') or self.vectorizer is None:
            return {}
        
        tfidf_names = list(self.vectorizer.get_feature_names_out())
        if self.use_multi_features:
            feature_names = tfidf_names + self.get_linguistic_feature_names()
        else:
            feature_names = tfidf_names
        importance_scores = np.abs(model.coef_[0])
        
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        
        feature_importance = {}
        for i in top_indices:
            if i < len(feature_names):
                feature_name = str(feature_names[i])
            else:
                feature_name = f'feature_{i}'
            importance_score = float(importance_scores[i])
            feature_importance[feature_name] = importance_score
        
        return feature_importance
