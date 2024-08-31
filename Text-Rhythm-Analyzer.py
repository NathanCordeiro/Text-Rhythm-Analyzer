import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
import numpy as np
from collections import Counter
import string
import spacy
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import csv
import os
from plotly.io import write_image, write_html
import pandas as pd

nltk.download('cmudict', quiet=True)
nltk.download('punkt', quiet=True)
nlp = spacy.load("en_core_web_sm")

class EnhancedTextRhythmAnalyzer:
    def __init__(self):
        self.cmu_dict = cmudict.dict()
        self.stress_patterns = {
            'iambic': '01',
            'trochaic': '10',
            'dactylic': '100',
            'anapestic': '001'
        }
        self.cache = {}

    def count_syllables(self, word):
        word = word.lower().strip(string.punctuation)
        if word in self.cache:
            return self.cache[word]
        if not word:  # Handle empty strings
            self.cache[word] = 0
            return 0
        if word in self.cmu_dict:
            count = max([len(list(y for y in x if y[-1].isdigit())) for x in self.cmu_dict[word]])
            self.cache[word] = count
            return count
        count = self._fallback_syllable_count(word)
        self.cache[word] = count
        return count

    def _fallback_syllable_count(self, word):
        if not word:  # Handle empty strings
            return 0
        count = 0
        vowels = 'aeiouy'
        word = word.lower().strip(string.punctuation)
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        return count

    def get_stress_pattern(self, word):
        word = word.lower().strip(string.punctuation)
        if word in self.cmu_dict:
            return ''.join([str(int(char[-1])) for char in self.cmu_dict[word][0] if char[-1].isdigit()])
        return '1' * self.count_syllables(word)  # Assume stress on all syllables if not found

    def identify_meter(self, stress_pattern):
        for name, pattern in self.stress_patterns.items():
            if stress_pattern.startswith(pattern):
                return name
        return 'irregular'

    def detect_phrase_breaks(self, text):
        doc = nlp(text)
        breaks = []
        for sent in doc.sents:
            breaks.append(sent.end)
        return breaks

    def analyze_text(self, text):
        sentences = sent_tokenize(text)
        words = [word for sent in sentences for word in word_tokenize(sent) if word.strip()]

        syllable_counts = [self.count_syllables(word) for word in words]
        stress_patterns = [self.get_stress_pattern(word) for word in words]

        # Rhythm analysis
        total_syllables = sum(syllable_counts)
        avg_syllables_per_word = total_syllables / len(words)

        # Meter analysis
        full_stress_pattern = ''.join(stress_patterns)
        meter = self.identify_meter(full_stress_pattern)

        # Prosody analysis
        phrase_breaks = self.detect_phrase_breaks(text)

        # Line analysis
        lines = text.split('\n')
        line_lengths = [len(line.split()) for line in lines if line.strip()]

        return {
            'words': words,
            'syllable_counts': syllable_counts,
            'stress_patterns': stress_patterns,
            'total_syllables': total_syllables,
            'avg_syllables_per_word': avg_syllables_per_word,
            'meter': meter,
            'phrase_breaks': phrase_breaks,
            'full_stress_pattern': full_stress_pattern,
            'line_lengths': line_lengths,
            'sentences': sentences
        }

    def visualize_rhythm_plotly(self, analysis_result):
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Syllable Count per Word',
            'Stress Pattern Visualization',
            'Distribution of Line Lengths',
            'Distribution of Sentence Lengths'
        ))

        # Syllable count graph
        fig.add_trace(go.Scatter(
            y=analysis_result['syllable_counts'],
            mode='lines',
            name='Syllable Count'
        ), row=1, col=1)

        # Stress pattern visualization
        max_len = max(len(pattern) for pattern in analysis_result['stress_patterns'])
        stress_matrix = np.zeros((len(analysis_result['stress_patterns']), max_len))
        for i, pattern in enumerate(analysis_result['stress_patterns']):
            for j, stress in enumerate(pattern):
                stress_matrix[i, j] = int(stress)

        fig.add_trace(go.Heatmap(
            z=stress_matrix,
            colorscale='YlOrRd',
            name='Stress Level'
        ), row=1, col=2)

        # Line length distribution
        fig.add_trace(go.Histogram(
            x=analysis_result['line_lengths'],
            name='Line Lengths'
        ), row=2, col=1)

        # Sentence length distribution
        sentence_lengths = [len(word_tokenize(sent)) for sent in analysis_result['sentences']]
        fig.add_trace(go.Histogram(
            x=sentence_lengths,
            name='Sentence Lengths'
        ), row=2, col=2)

        fig.update_layout(height=800, width=1000, title_text="Text Rhythm Analysis")
        fig.show()

    def print_analysis(self, text):
        analysis = self.analyze_text(text)
        print(f"Total syllables: {analysis['total_syllables']}")
        print(f"Average syllables per word: {analysis['avg_syllables_per_word']:.2f}")
        print(f"Detected meter: {analysis['meter']}")
        print(f"Phrase breaks at word positions: {analysis['phrase_breaks']}")

        print("\nWord Analysis (first 10 words):")
        for i, word in enumerate(analysis['words'][:10]):
            print(f"{word}: Syllables: {analysis['syllable_counts'][i]}, Stress Pattern: {analysis['stress_patterns'][i]}")

        print("\nFull Stress Pattern:")
        print(analysis['full_stress_pattern'])

        print("\nLine Lengths:")
        print(analysis['line_lengths'])

        print(f"\nMost common stress patterns:")
        stress_counter = Counter(analysis['stress_patterns'])
        for pattern, count in stress_counter.most_common(5):
            print(f"  {pattern}: {count} occurrences")

        avg_line_length = np.mean(analysis['line_lengths'])
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in analysis['sentences']])
        print(f"\nAverage line length: {avg_line_length:.2f} words")
        print(f"Average sentence length: {avg_sentence_length:.2f} words")

        # Visualize the analysis results using Plotly
        self.visualize_rhythm_plotly(analysis)
        
    def export_analysis_to_csv(self, analysis_result, filename='text_rhythm_analysis.csv'):
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total syllables', analysis_result['total_syllables']])
                writer.writerow(['Average syllables per word', f"{analysis_result['avg_syllables_per_word']:.2f}"])
                writer.writerow(['Detected meter', analysis_result['meter']])
                writer.writerow(['Phrase breaks at word positions', ', '.join(map(str, analysis_result['phrase_breaks']))])
                
                writer.writerow([])
                writer.writerow(['Word', 'Syllables', 'Stress Pattern'])
                for i, word in enumerate(analysis_result['words']):
                    writer.writerow([word, analysis_result['syllable_counts'][i], analysis_result['stress_patterns'][i]])
                
                writer.writerow([])
                writer.writerow(['Full Stress Pattern'])
                writer.writerow([analysis_result['full_stress_pattern']])
                
                writer.writerow([])
                writer.writerow(['Line Lengths'])
                writer.writerow(analysis_result['line_lengths'])
                
                writer.writerow([])
                writer.writerow(['Most common stress patterns'])
                stress_counter = Counter(analysis_result['stress_patterns'])
                for pattern, count in stress_counter.most_common(5):
                    writer.writerow([pattern, count])
                
                writer.writerow([])
                writer.writerow(['Average line length', f"{np.mean(analysis_result['line_lengths']):.2f}"])
                writer.writerow(['Average sentence length', f"{np.mean([len(word_tokenize(sent)) for sent in analysis_result['sentences']]):.2f}"])

            print(f"Analysis results exported to {filename}")

    def export_visualizations(self, analysis_result, base_filename='text_rhythm_visualization'):
        fig = self.create_visualization(analysis_result)
        
        # Export as interactive HTML
        html_filename = f"{base_filename}.html"
        write_html(fig, html_filename)
        print(f"Interactive visualization exported to {html_filename}")
        
        # Export as static PDF
        pdf_filename = f"{base_filename}.pdf"
        write_image(fig, pdf_filename)
        print(f"Static visualization exported to {pdf_filename}")

    def create_visualization(self, analysis_result):
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Syllable Count per Word',
            'Stress Pattern Visualization',
            'Distribution of Line Lengths',
            'Distribution of Sentence Lengths'
        ))

        # Syllable count graph
        fig.add_trace(go.Scatter(
            y=analysis_result['syllable_counts'],
            mode='lines',
            name='Syllable Count'
        ), row=1, col=1)

        # Stress pattern visualization
        max_len = max(len(pattern) for pattern in analysis_result['stress_patterns'])
        stress_matrix = np.zeros((len(analysis_result['stress_patterns']), max_len))
        for i, pattern in enumerate(analysis_result['stress_patterns']):
            for j, stress in enumerate(pattern):
                stress_matrix[i, j] = int(stress)

        fig.add_trace(go.Heatmap(
            z=stress_matrix,
            colorscale='YlOrRd',
            name='Stress Level'
        ), row=1, col=2)

        # Line length distribution
        fig.add_trace(go.Histogram(
            x=analysis_result['line_lengths'],
            name='Line Lengths'
        ), row=2, col=1)

        # Sentence length distribution
        sentence_lengths = [len(word_tokenize(sent)) for sent in analysis_result['sentences']]
        fig.add_trace(go.Histogram(
            x=sentence_lengths,
            name='Sentence Lengths'
        ), row=2, col=2)

        fig.update_layout(height=800, width=1000, title_text="Text Rhythm Analysis")
        return fig

    def analyze_and_export(self, text, export_dir='exports'):
        analysis_result = self.analyze_text(text)
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Export analysis results to CSV
        csv_filename = os.path.join(export_dir, 'text_rhythm_analysis.csv')
        self.export_analysis_to_csv(analysis_result, csv_filename)
        
        # Export visualizations
        base_filename = os.path.join(export_dir, 'text_rhythm_visualization')
        self.export_visualizations(analysis_result, base_filename)
        
        # Display results and interactive visualization
        self.print_analysis(analysis_result)
        fig = self.create_visualization(analysis_result)
        fig.show()

# Example usage
if __name__ == "__main__":
    analyzer = EnhancedTextRhythmAnalyzer()

    sample_text = """
    Shall I compare thee to a summer's day?
    Thou art more lovely and more temperate:
    Rough winds do shake the darling buds of May,
    And summer's lease hath all too short a date;
    Sometime too hot the eye of heaven shines,
    And often is his gold complexion dimmed;
    And every fair from fair sometime declines,
    By chance, or nature's changing course, untrimmed;
    But thy eternal summer shall not fade,
    Nor lose possession of that fair thou ow'st,
    Nor shall death brag thou wand'rest in his shade,
    When in eternal lines to Time thou grow'st.
    So long as men can breathe, or eyes can see,
    So long lives this, and this gives life to thee.
    """

    analyzer.analyze_and_export(sample_text)