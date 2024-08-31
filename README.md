# Text-Rhythm-Analyzer

## Overview
The Enhanced Text Rhythm Analyzer is a Python-based tool designed to analyze the rhythmic structure and prosodic features of a given text. It uses various natural language processing (NLP) techniques to identify syllable counts, stress patterns, natural and unnatural pauses, prosodic phrasing, and fluency/disfluency patterns in the text. The tool also provides visualization capabilities to better understand the rhythmic and prosodic characteristics of the analyzed text.

## Features

| Feature                           | Description                                                                                       |
|-----------------------------------|---------------------------------------------------------------------------------------------------|
| Syllable Counting                 | Counts the number of syllables in each word using CMU Pronouncing Dictionary and fallback methods. |
| Stress Pattern Detection          | Identifies stress patterns (e.g., iambic, trochaic) for each word and aggregates for full text.    |
| Meter Identification              | Detects poetic meter based on identified stress patterns.                                          |
| Prosody Analysis                  | Analyzes phrase breaks, prosodic phrasing, and fluency/disfluency patterns.                       |
| Sentence and Line Length Analysis | Provides statistical analysis of sentence and line lengths.                                        |
| Visualization                     | Visualizes syllable counts, stress patterns, and sentence/line length distributions using Plotly.  |
| Caching Mechanism                 | Uses a caching mechanism to store syllable counts for faster processing of repeated words.         |
| Export Analysis                   | Allows exporting of analysis results to CSV and visualizations to image files or HTML.             |

## Requirements
To run the Enhanced Text Rhythm Analyzer, you'll need the following Python libraries:
```bash
pip install nltk spacy plotly kaleido pandas
python -m spacy download en_core_web_sm
python -m nltk.downloader cmudict punkt
```

## Setup and Usage
1. Clone the Repository
   ```bash
   git clone https://github.com/NathanCordeiro/Text-Rhythm-Analyzer.git
   cd Text-Rhythm-Analyzer
   ```
2. Run the Script
   ```
   python Text-Rhythm-Analyzer.py
   ```
3. Analyze Custom Text
   ```
   sample_text = """
   Your custom text here.
   """

   analyzer.analyze_and_export(sample_text)
   ```

