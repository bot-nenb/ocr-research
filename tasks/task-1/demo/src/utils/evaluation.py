"""
Evaluation Metrics for OCR Performance

This module provides comprehensive evaluation metrics including WER, CER,
and other accuracy measurements.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import Levenshtein
import numpy as np
from scipy import stats


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    doc_id: str
    wer: float
    cer: float
    word_accuracy: float
    char_accuracy: float
    edit_distance: int
    num_words_ref: int
    num_words_hyp: int
    num_chars_ref: int
    num_chars_hyp: int
    confidence_score: float = 0.0


class TextNormalizer:
    """Normalize text for fair comparison."""
    
    @staticmethod
    def normalize(text: str, lowercase: bool = True, 
                 remove_punctuation: bool = False,
                 collapse_whitespace: bool = True) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            collapse_whitespace: Collapse multiple spaces to single space
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove unicode artifacts
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove punctuation
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Collapse whitespace
        if collapse_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        normalized = TextNormalizer.normalize(text)
        return normalized.split() if normalized else []


class EvaluationMetrics:
    """Calculate OCR evaluation metrics."""
    
    def __init__(self, normalize_text: bool = True,
                 lowercase: bool = True,
                 remove_punctuation: bool = False):
        """
        Initialize evaluation metrics calculator.
        
        Args:
            normalize_text: Whether to normalize text before comparison
            lowercase: Convert to lowercase for comparison
            remove_punctuation: Remove punctuation for comparison
        """
        self.normalize_text = normalize_text
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.results: List[EvaluationResult] = []
    
    def _sequence_edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Calculate edit distance between two sequences using dynamic programming.
        
        Args:
            seq1: Reference sequence (list of words)
            seq2: Hypothesis sequence (list of words)
            
        Returns:
            Edit distance between sequences
        """
        len1, len2 = len(seq1), len(seq2)
        
        # Initialize DP table
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )
        
        return dp[len1][len2]
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER).
        
        WER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = words in reference
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            
        Returns:
            WER score (0.0 = perfect, 1.0+ = poor)
        """
        if self.normalize_text:
            reference = TextNormalizer.normalize(
                reference, self.lowercase, self.remove_punctuation
            )
            hypothesis = TextNormalizer.normalize(
                hypothesis, self.lowercase, self.remove_punctuation
            )
        
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        
        # Calculate edit distance at word level using sequence edit distance
        distance = self._sequence_edit_distance(ref_words, hyp_words)
        wer = distance / len(ref_words)
        
        return min(wer, 1.0)  # Cap at 1.0 for percentage calculations
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER).
        
        CER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = chars in reference
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            
        Returns:
            CER score (0.0 = perfect, 1.0+ = poor)
        """
        if self.normalize_text:
            reference = TextNormalizer.normalize(
                reference, self.lowercase, self.remove_punctuation
            )
            hypothesis = TextNormalizer.normalize(
                hypothesis, self.lowercase, self.remove_punctuation
            )
        
        if not reference:
            return 0.0 if not hypothesis else 1.0
        
        # Calculate edit distance at character level
        distance = Levenshtein.distance(reference, hypothesis)
        cer = distance / len(reference)
        
        return min(cer, 1.0)  # Cap at 1.0 for percentage calculations
    
    def calculate_accuracy(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """
        Calculate word and character accuracy.
        
        Args:
            reference: Ground truth text
            hypothesis: OCR output text
            
        Returns:
            Tuple of (word_accuracy, char_accuracy)
        """
        wer = self.calculate_wer(reference, hypothesis)
        cer = self.calculate_cer(reference, hypothesis)
        
        word_accuracy = max(0, 1 - wer)
        char_accuracy = max(0, 1 - cer)
        
        return word_accuracy, char_accuracy
    
    def evaluate_document(self, doc_id: str, reference: str, 
                         hypothesis: str, confidence: float = 0.0) -> EvaluationResult:
        """
        Evaluate a single document.
        
        Args:
            doc_id: Document identifier
            reference: Ground truth text
            hypothesis: OCR output text
            confidence: OCR confidence score
            
        Returns:
            EvaluationResult object
        """
        # Normalize for metrics calculation
        if self.normalize_text:
            ref_norm = TextNormalizer.normalize(
                reference, self.lowercase, self.remove_punctuation
            )
            hyp_norm = TextNormalizer.normalize(
                hypothesis, self.lowercase, self.remove_punctuation
            )
        else:
            ref_norm = reference
            hyp_norm = hypothesis
        
        # Calculate metrics
        wer = self.calculate_wer(reference, hypothesis)
        cer = self.calculate_cer(reference, hypothesis)
        word_acc, char_acc = self.calculate_accuracy(reference, hypothesis)
        
        # Get counts
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        
        result = EvaluationResult(
            doc_id=doc_id,
            wer=wer,
            cer=cer,
            word_accuracy=word_acc,
            char_accuracy=char_acc,
            edit_distance=Levenshtein.distance(ref_norm, hyp_norm),
            num_words_ref=len(ref_words),
            num_words_hyp=len(hyp_words),
            num_chars_ref=len(ref_norm),
            num_chars_hyp=len(hyp_norm),
            confidence_score=confidence
        )
        
        self.results.append(result)
        return result
    
    def evaluate_batch(self, documents: List[Tuple[str, str, str]], 
                       confidences: Optional[List[float]] = None) -> List[EvaluationResult]:
        """
        Evaluate a batch of documents.
        
        Args:
            documents: List of (doc_id, reference, hypothesis) tuples
            confidences: Optional list of confidence scores
            
        Returns:
            List of evaluation results
        """
        if confidences is None:
            confidences = [0.0] * len(documents)
        
        results = []
        for (doc_id, ref, hyp), conf in zip(documents, confidences):
            result = self.evaluate_document(doc_id, ref, hyp, conf)
            results.append(result)
        
        return results
    
    def get_aggregate_metrics(self) -> Dict:
        """
        Get aggregate metrics across all evaluated documents.
        
        Returns:
            Dictionary of aggregate metrics
        """
        if not self.results:
            return {}
        
        wer_scores = [r.wer for r in self.results]
        cer_scores = [r.cer for r in self.results]
        word_acc_scores = [r.word_accuracy for r in self.results]
        char_acc_scores = [r.char_accuracy for r in self.results]
        confidence_scores = [r.confidence_score for r in self.results if r.confidence_score > 0]
        
        metrics = {
            "num_documents": len(self.results),
            "wer": {
                "mean": np.mean(wer_scores),
                "std": np.std(wer_scores),
                "min": np.min(wer_scores),
                "max": np.max(wer_scores),
                "median": np.median(wer_scores),
                "percentile_25": np.percentile(wer_scores, 25),
                "percentile_75": np.percentile(wer_scores, 75)
            },
            "cer": {
                "mean": np.mean(cer_scores),
                "std": np.std(cer_scores),
                "min": np.min(cer_scores),
                "max": np.max(cer_scores),
                "median": np.median(cer_scores),
                "percentile_25": np.percentile(cer_scores, 25),
                "percentile_75": np.percentile(cer_scores, 75)
            },
            "word_accuracy": {
                "mean": np.mean(word_acc_scores),
                "std": np.std(word_acc_scores),
                "min": np.min(word_acc_scores),
                "max": np.max(word_acc_scores),
                "median": np.median(word_acc_scores)
            },
            "char_accuracy": {
                "mean": np.mean(char_acc_scores),
                "std": np.std(char_acc_scores),
                "min": np.min(char_acc_scores),
                "max": np.max(char_acc_scores),
                "median": np.median(char_acc_scores)
            },
            "total_words_reference": sum(r.num_words_ref for r in self.results),
            "total_words_hypothesis": sum(r.num_words_hyp for r in self.results),
            "total_chars_reference": sum(r.num_chars_ref for r in self.results),
            "total_chars_hypothesis": sum(r.num_chars_hyp for r in self.results)
        }
        
        if confidence_scores:
            metrics["confidence"] = {
                "mean": np.mean(confidence_scores),
                "std": np.std(confidence_scores),
                "min": np.min(confidence_scores),
                "max": np.max(confidence_scores)
            }
        
        # Calculate accuracy thresholds
        accuracy_thresholds = [0.9, 0.95, 0.99]
        for threshold in accuracy_thresholds:
            metrics[f"docs_above_{int(threshold*100)}_word_acc"] = sum(
                1 for r in self.results if r.word_accuracy >= threshold
            )
            metrics[f"docs_above_{int(threshold*100)}_char_acc"] = sum(
                1 for r in self.results if r.char_accuracy >= threshold
            )
        
        return metrics
    
    def get_correlation_analysis(self) -> Dict:
        """
        Analyze correlations between metrics.
        
        Returns:
            Dictionary with correlation analysis
        """
        if len(self.results) < 3:
            return {}
        
        confidence_scores = [r.confidence_score for r in self.results if r.confidence_score > 0]
        
        if len(confidence_scores) < 3:
            return {}
        
        wer_scores = [r.wer for r in self.results if r.confidence_score > 0]
        cer_scores = [r.cer for r in self.results if r.confidence_score > 0]
        
        analysis = {}
        
        # Correlation between confidence and accuracy
        if confidence_scores:
            wer_corr, wer_p = stats.pearsonr(confidence_scores, wer_scores)
            cer_corr, cer_p = stats.pearsonr(confidence_scores, cer_scores)
            
            analysis["confidence_wer_correlation"] = {
                "correlation": wer_corr,
                "p_value": wer_p,
                "significant": wer_p < 0.05
            }
            
            analysis["confidence_cer_correlation"] = {
                "correlation": cer_corr,
                "p_value": cer_p,
                "significant": cer_p < 0.05
            }
        
        # Document length effect
        doc_lengths = [r.num_chars_ref for r in self.results]
        wer_all = [r.wer for r in self.results]
        
        length_corr, length_p = stats.pearsonr(doc_lengths, wer_all)
        analysis["length_wer_correlation"] = {
            "correlation": length_corr,
            "p_value": length_p,
            "significant": length_p < 0.05
        }
        
        return analysis
    
    def get_worst_performers(self, n: int = 10) -> List[EvaluationResult]:
        """
        Get the worst performing documents.
        
        Args:
            n: Number of worst performers to return
            
        Returns:
            List of worst performing documents by WER
        """
        sorted_results = sorted(self.results, key=lambda x: x.wer, reverse=True)
        return sorted_results[:n]
    
    def get_best_performers(self, n: int = 10) -> List[EvaluationResult]:
        """
        Get the best performing documents.
        
        Args:
            n: Number of best performers to return
            
        Returns:
            List of best performing documents by WER
        """
        sorted_results = sorted(self.results, key=lambda x: x.wer)
        return sorted_results[:n]
    
    def print_summary(self):
        """Print evaluation summary to console."""
        metrics = self.get_aggregate_metrics()
        
        if not metrics:
            print("No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("OCR EVALUATION SUMMARY")
        print("="*60)
        print(f"Documents evaluated: {metrics['num_documents']}")
        print("\nWord Error Rate (WER):")
        print(f"  Mean: {metrics['wer']['mean']:.3f}")
        print(f"  Std:  {metrics['wer']['std']:.3f}")
        print(f"  Range: [{metrics['wer']['min']:.3f}, {metrics['wer']['max']:.3f}]")
        
        print("\nCharacter Error Rate (CER):")
        print(f"  Mean: {metrics['cer']['mean']:.3f}")
        print(f"  Std:  {metrics['cer']['std']:.3f}")
        print(f"  Range: [{metrics['cer']['min']:.3f}, {metrics['cer']['max']:.3f}]")
        
        print("\nAccuracy:")
        print(f"  Word Accuracy: {metrics['word_accuracy']['mean']*100:.1f}%")
        print(f"  Char Accuracy: {metrics['char_accuracy']['mean']*100:.1f}%")
        
        print("\nQuality Distribution:")
        for threshold in [90, 95, 99]:
            word_count = metrics.get(f'docs_above_{threshold}_word_acc', 0)
            char_count = metrics.get(f'docs_above_{threshold}_char_acc', 0)
            print(f"  >{threshold}% word accuracy: {word_count}/{metrics['num_documents']} documents")
            print(f"  >{threshold}% char accuracy: {char_count}/{metrics['num_documents']} documents")
        
        if 'confidence' in metrics:
            print(f"\nConfidence Score:")
            print(f"  Mean: {metrics['confidence']['mean']:.3f}")
            print(f"  Range: [{metrics['confidence']['min']:.3f}, {metrics['confidence']['max']:.3f}]")
        
        print("="*60)