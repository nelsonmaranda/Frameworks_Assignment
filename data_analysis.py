"""
Data Analysis and Visualization for CORD-19 Assignment
Part 3: Data Analysis and Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import os
from datetime import datetime

def load_cleaned_data(file_path='data/sample_metadata.csv'):
    """
    Load and perform basic cleaning on the dataset.
    
    Args:
        file_path (str): Path to the metadata CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Loading and cleaning data...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert publish_time to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    
    # Create abstract word count
    df['abstract_word_count'] = df['abstract'].str.split().str.len()
    
    # Create title word count
    df['title_word_count'] = df['title'].str.split().str.len()
    
    # Clean journal names (remove extra spaces, standardize)
    df['journal'] = df['journal'].str.strip()
    
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    return df

def analyze_publication_trends(df):
    """
    Analyze publication trends over time.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Analysis results
    """
    print("\nAnalyzing publication trends...")
    
    # Filter out missing years
    df_clean = df.dropna(subset=['year'])
    
    # Yearly publication counts
    yearly_counts = df_clean['year'].value_counts().sort_index()
    
    # Monthly publication counts (for recent years)
    df_clean['month'] = df_clean['publish_time'].dt.month
    df_clean['year_month'] = df_clean['publish_time'].dt.to_period('M')
    monthly_counts = df_clean['year_month'].value_counts().sort_index()
    
    # Calculate growth rates
    growth_rates = yearly_counts.pct_change() * 100
    
    results = {
        'yearly_counts': yearly_counts,
        'monthly_counts': monthly_counts,
        'growth_rates': growth_rates,
        'total_publications': len(df_clean),
        'date_range': (df_clean['publish_time'].min(), df_clean['publish_time'].max())
    }
    
    print(f"Total publications: {results['total_publications']:,}")
    print(f"Date range: {results['date_range'][0]} to {results['date_range'][1]}")
    print(f"Peak year: {yearly_counts.idxmax()} with {yearly_counts.max()} publications")
    
    return results

def analyze_journal_patterns(df):
    """
    Analyze journal publication patterns.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Analysis results
    """
    print("\nAnalyzing journal patterns...")
    
    # Top journals
    journal_counts = df['journal'].value_counts()
    top_journals = journal_counts.head(20)
    
    # Journal diversity metrics
    total_journals = len(journal_counts)
    gini_coefficient = calculate_gini_coefficient(journal_counts.values)
    
    # Publications per journal statistics
    journal_stats = {
        'mean_pubs_per_journal': journal_counts.mean(),
        'median_pubs_per_journal': journal_counts.median(),
        'std_pubs_per_journal': journal_counts.std(),
        'max_pubs_per_journal': journal_counts.max()
    }
    
    # Journal concentration (top 10 journals)
    top_10_share = (journal_counts.head(10).sum() / journal_counts.sum()) * 100
    
    results = {
        'journal_counts': journal_counts,
        'top_journals': top_journals,
        'total_journals': total_journals,
        'gini_coefficient': gini_coefficient,
        'journal_stats': journal_stats,
        'top_10_share': top_10_share
    }
    
    print(f"Total journals: {total_journals}")
    print(f"Top 10 journals account for {top_10_share:.1f}% of publications")
    print(f"Gini coefficient (inequality): {gini_coefficient:.3f}")
    
    return results

def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient for inequality measurement.
    
    Args:
        values (array-like): Values to calculate Gini coefficient for
        
    Returns:
        float: Gini coefficient
    """
    values = np.array(values)
    n = len(values)
    if n == 0:
        return 0
    
    # Sort values
    sorted_values = np.sort(values)
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def analyze_title_content(df):
    """
    Analyze title content and word frequency.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Analysis results
    """
    print("\nAnalyzing title content...")
    
    # Combine all titles
    all_titles = ' '.join(df['title'].dropna().astype(str))
    
    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they',
        'have', 'has', 'had', 'was', 'were', 'been', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'must', 'shall', 'not', 'but', 'or',
        'nor', 'yet', 'so', 'if', 'then', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'only', 'own', 'same', 'than', 'too', 'very', 'just', 'now', 'also',
        'new', 'first', 'last', 'long', 'great', 'little', 'good', 'bad',
        'high', 'low', 'large', 'small', 'big', 'old', 'young', 'early', 'late'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Word frequency
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(30)
    
    # Title length analysis
    title_lengths = df['title'].str.len()
    
    results = {
        'word_counts': word_counts,
        'top_words': top_words,
        'total_words': len(filtered_words),
        'unique_words': len(word_counts),
        'title_length_stats': {
            'mean': title_lengths.mean(),
            'median': title_lengths.median(),
            'std': title_lengths.std(),
            'min': title_lengths.min(),
            'max': title_lengths.max()
        }
    }
    
    print(f"Total words analyzed: {results['total_words']:,}")
    print(f"Unique words: {results['unique_words']:,}")
    print(f"Average title length: {results['title_length_stats']['mean']:.1f} characters")
    
    return results

def analyze_abstract_content(df):
    """
    Analyze abstract content.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Analysis results
    """
    print("\nAnalyzing abstract content...")
    
    # Filter papers with abstracts
    df_with_abstracts = df.dropna(subset=['abstract'])
    
    if len(df_with_abstracts) == 0:
        print("No abstracts found in dataset")
        return {}
    
    # Abstract length analysis
    abstract_lengths = df_with_abstracts['abstract'].str.len()
    abstract_word_counts = df_with_abstracts['abstract_word_count']
    
    # Combine all abstracts
    all_abstracts = ' '.join(df_with_abstracts['abstract'].astype(str))
    
    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_abstracts.lower())
    
    # Remove stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they',
        'have', 'has', 'had', 'was', 'were', 'been', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'must', 'shall', 'not', 'but', 'or',
        'nor', 'yet', 'so', 'if', 'then', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'only', 'own', 'same', 'than', 'too', 'very', 'just', 'now', 'also',
        'new', 'first', 'last', 'long', 'great', 'little', 'good', 'bad',
        'high', 'low', 'large', 'small', 'big', 'old', 'young', 'early', 'late',
        'study', 'studies', 'research', 'analysis', 'results', 'findings',
        'conclusion', 'conclusions', 'method', 'methods', 'data', 'patients',
        'cases', 'control', 'group', 'groups', 'treatment', 'treatments'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(30)
    
    results = {
        'papers_with_abstracts': len(df_with_abstracts),
        'abstract_length_stats': {
            'mean': abstract_lengths.mean(),
            'median': abstract_lengths.median(),
            'std': abstract_lengths.std(),
            'min': abstract_lengths.min(),
            'max': abstract_lengths.max()
        },
        'abstract_word_stats': {
            'mean': abstract_word_counts.mean(),
            'median': abstract_word_counts.median(),
            'std': abstract_word_counts.std(),
            'min': abstract_word_counts.min(),
            'max': abstract_word_counts.max()
        },
        'word_counts': word_counts,
        'top_words': top_words
    }
    
    print(f"Papers with abstracts: {results['papers_with_abstracts']:,}")
    print(f"Average abstract length: {results['abstract_length_stats']['mean']:.0f} characters")
    print(f"Average abstract word count: {results['abstract_word_stats']['mean']:.0f} words")
    
    return results

def create_publication_visualizations(trend_results, journal_results):
    """
    Create visualizations for publication trends and journal analysis.
    
    Args:
        trend_results (dict): Publication trend analysis results
        journal_results (dict): Journal analysis results
    """
    print("\nCreating publication visualizations...")
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Publications over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Yearly trends
    yearly_counts = trend_results['yearly_counts']
    ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)
    ax1.set_title('COVID-19 Research Publications by Year', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Publications')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on points
    for i, v in enumerate(yearly_counts.values):
        ax1.annotate(f'{v:,}', (yearly_counts.index[i], v), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Monthly trends (last 2 years)
    monthly_counts = trend_results['monthly_counts']
    recent_months = monthly_counts[monthly_counts.index >= '2020-01']
    ax2.plot(range(len(recent_months)), recent_months.values, marker='o', linewidth=2)
    ax2.set_title('Monthly Publication Trends (2020-2022)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Publications')
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels
    month_labels = [str(period) for period in recent_months.index[::3]]
    ax2.set_xticks(range(0, len(recent_months), 3))
    ax2.set_xticklabels(month_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/publication_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Top journals
    plt.figure(figsize=(14, 10))
    top_journals = journal_results['top_journals'].head(15)
    bars = plt.barh(range(len(top_journals)), top_journals.values)
    plt.yticks(range(len(top_journals)), top_journals.index)
    plt.xlabel('Number of Publications')
    plt.title('Top 15 Journals by Publication Count', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, v in enumerate(top_journals.values):
        plt.text(v + 0.5, i, f'{v:,}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/top_journals_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Journal concentration
    plt.figure(figsize=(10, 8))
    journal_counts = journal_results['journal_counts']
    
    # Calculate cumulative percentage
    sorted_counts = journal_counts.sort_values(ascending=False)
    cumulative_pct = (sorted_counts.cumsum() / sorted_counts.sum()) * 100
    
    plt.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, linewidth=2)
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    plt.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    plt.xlabel('Number of Journals (ranked by publications)')
    plt.ylabel('Cumulative Percentage of Publications')
    plt.title('Journal Concentration: Cumulative Share of Publications', 
              fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/journal_concentration.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_content_visualizations(title_results, abstract_results):
    """
    Create visualizations for content analysis.
    
    Args:
        title_results (dict): Title analysis results
        abstract_results (dict): Abstract analysis results
    """
    print("\nCreating content visualizations...")
    
    # 1. Word frequency in titles
    plt.figure(figsize=(14, 8))
    top_words = title_results['top_words'][:20]
    words, counts = zip(*top_words)
    
    bars = plt.barh(range(len(words)), counts)
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency')
    plt.title('Top 20 Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(counts):
        plt.text(v + 0.5, i, f'{v:,}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/title_word_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Title length distribution
    plt.figure(figsize=(12, 6))
    plt.hist(title_results['title_length_stats'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(title_results['title_length_stats']['mean'], color='red', 
                linestyle='--', linewidth=2, label=f"Mean: {title_results['title_length_stats']['mean']:.1f}")
    plt.axvline(title_results['title_length_stats']['median'], color='green', 
                linestyle='--', linewidth=2, label=f"Median: {title_results['title_length_stats']['median']:.1f}")
    plt.xlabel('Title Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Title Lengths', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/title_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Word cloud for titles
    if title_results['word_counts']:
        plt.figure(figsize=(15, 8))
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100,
                            colormap='viridis').generate_from_frequencies(title_results['word_counts'])
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: Most Frequent Words in Paper Titles', 
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/title_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Abstract analysis (if available)
    if abstract_results and 'abstract_length_stats' in abstract_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Abstract length distribution
        ax1.hist(abstract_results['abstract_length_stats'], bins=30, 
                edgecolor='black', alpha=0.7)
        ax1.axvline(abstract_results['abstract_length_stats']['mean'], 
                   color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {abstract_results['abstract_length_stats']['mean']:.0f}")
        ax1.set_xlabel('Abstract Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Abstract Lengths', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Abstract word count distribution
        ax2.hist(abstract_results['abstract_word_stats'], bins=30, 
                edgecolor='black', alpha=0.7)
        ax2.axvline(abstract_results['abstract_word_stats']['mean'], 
                   color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {abstract_results['abstract_word_stats']['mean']:.0f}")
        ax2.set_xlabel('Abstract Word Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Abstract Word Counts', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/abstract_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_analysis_report(trend_results, journal_results, title_results, abstract_results):
    """
    Generate a comprehensive analysis report.
    
    Args:
        trend_results (dict): Publication trend analysis results
        journal_results (dict): Journal analysis results
        title_results (dict): Title analysis results
        abstract_results (dict): Abstract analysis results
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    
    # Publication trends summary
    print("\n📊 PUBLICATION TRENDS:")
    print(f"   • Total publications analyzed: {trend_results['total_publications']:,}")
    print(f"   • Date range: {trend_results['date_range'][0].strftime('%Y-%m-%d')} to {trend_results['date_range'][1].strftime('%Y-%m-%d')}")
    print(f"   • Peak year: {trend_results['yearly_counts'].idxmax()} with {trend_results['yearly_counts'].max():,} publications")
    
    # Journal analysis summary
    print(f"\n📚 JOURNAL ANALYSIS:")
    print(f"   • Total journals: {journal_results['total_journals']:,}")
    print(f"   • Top 10 journals account for {journal_results['top_10_share']:.1f}% of publications")
    print(f"   • Gini coefficient (inequality): {journal_results['gini_coefficient']:.3f}")
    print(f"   • Most prolific journal: {journal_results['top_journals'].index[0]} ({journal_results['top_journals'].iloc[0]:,} papers)")
    
    # Title analysis summary
    print(f"\n📝 TITLE ANALYSIS:")
    print(f"   • Average title length: {title_results['title_length_stats']['mean']:.1f} characters")
    print(f"   • Total words analyzed: {title_results['total_words']:,}")
    print(f"   • Unique words: {title_results['unique_words']:,}")
    print(f"   • Most common word: '{title_results['top_words'][0][0]}' ({title_results['top_words'][0][1]:,} occurrences)")
    
    # Abstract analysis summary
    if abstract_results and 'papers_with_abstracts' in abstract_results:
        print(f"\n📄 ABSTRACT ANALYSIS:")
        print(f"   • Papers with abstracts: {abstract_results['papers_with_abstracts']:,}")
        print(f"   • Average abstract length: {abstract_results['abstract_length_stats']['mean']:.0f} characters")
        print(f"   • Average abstract word count: {abstract_results['abstract_word_stats']['mean']:.0f} words")
    
    print(f"\n📈 KEY INSIGHTS:")
    print(f"   • The dataset shows significant research activity in COVID-19 related topics")
    print(f"   • Publication patterns indicate rapid response to the pandemic")
    print(f"   • Journal distribution shows both concentration and diversity")
    print(f"   • Title analysis reveals common research themes and terminology")
    
    print(f"\n📁 All visualizations saved to 'plots/' directory")

def main():
    """Main function to run data analysis."""
    print("CORD-19 Data Analysis and Visualization")
    print("=" * 60)
    
    # Load and clean data
    df = load_cleaned_data()
    
    # Perform analyses
    trend_results = analyze_publication_trends(df)
    journal_results = analyze_journal_patterns(df)
    title_results = analyze_title_content(df)
    abstract_results = analyze_abstract_content(df)
    
    # Create visualizations
    create_publication_visualizations(trend_results, journal_results)
    create_content_visualizations(title_results, abstract_results)
    
    # Generate report
    generate_analysis_report(trend_results, journal_results, title_results, abstract_results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Next step: Run streamlit_app.py for interactive dashboard")

if __name__ == "__main__":
    main()
