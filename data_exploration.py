"""
Data Loading and Basic Exploration for CORD-19 Assignment
Part 1: Data Loading and Basic Exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_data(file_path='data/sample_metadata.csv'):
    """
    Load the CORD-19 metadata dataset.
    
    Args:
        file_path (str): Path to the metadata CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please run sample_data_generator.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_exploration(df):
    """
    Perform basic data exploration.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    print("=" * 50)
    print("BASIC DATA EXPLORATION")
    print("=" * 50)
    
    # 1. DataFrame dimensions
    print(f"\n1. Dataset Dimensions:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    # 2. Column information
    print(f"\n2. Column Information:")
    print(f"   Column names: {list(df.columns)}")
    
    # 3. Data types
    print(f"\n3. Data Types:")
    print(df.dtypes)
    
    # 4. First few rows
    print(f"\n4. First 5 rows:")
    print(df.head())
    
    # 5. Basic statistics for numerical columns
    print(f"\n5. Basic Statistics:")
    print(df.describe(include='all'))
    
    # 6. Missing values analysis
    print(f"\n6. Missing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percent.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    return missing_df

def explore_publication_dates(df):
    """
    Explore publication date patterns.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    print("\n" + "=" * 50)
    print("PUBLICATION DATE ANALYSIS")
    print("=" * 50)
    
    # Convert publish_time to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    
    # Extract year
    df['year'] = df['publish_time'].dt.year
    
    # Publication year distribution
    print(f"\n1. Publication Year Distribution:")
    year_counts = df['year'].value_counts().sort_index()
    print(year_counts)
    
    # Date range
    print(f"\n2. Date Range:")
    print(f"   Earliest: {df['publish_time'].min()}")
    print(f"   Latest: {df['publish_time'].max()}")
    
    # Missing dates
    missing_dates = df['publish_time'].isnull().sum()
    print(f"   Missing dates: {missing_dates} ({missing_dates/len(df)*100:.1f}%)")
    
    return year_counts

def explore_journals(df):
    """
    Explore journal information.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    print("\n" + "=" * 50)
    print("JOURNAL ANALYSIS")
    print("=" * 50)
    
    # Journal distribution
    print(f"\n1. Journal Distribution:")
    journal_counts = df['journal'].value_counts()
    print(f"   Total unique journals: {len(journal_counts)}")
    print(f"\n   Top 10 journals:")
    print(journal_counts.head(10))
    
    # Missing journal data
    missing_journals = df['journal'].isnull().sum()
    print(f"\n2. Missing journal data: {missing_journals} ({missing_journals/len(df)*100:.1f}%)")
    
    return journal_counts

def explore_authors(df):
    """
    Explore author information.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    print("\n" + "=" * 50)
    print("AUTHOR ANALYSIS")
    print("=" * 50)
    
    # Author information
    print(f"\n1. Author Information:")
    missing_authors = df['authors'].isnull().sum()
    print(f"   Papers with authors: {len(df) - missing_authors}")
    print(f"   Missing author data: {missing_authors} ({missing_authors/len(df)*100:.1f}%)")
    
    # Papers with multiple authors
    df_with_authors = df.dropna(subset=['authors'])
    multiple_authors = df_with_authors['authors'].str.contains(';').sum()
    print(f"   Papers with multiple authors: {multiple_authors}")
    
    return df_with_authors

def explore_titles(df):
    """
    Explore title information.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    print("\n" + "=" * 50)
    print("TITLE ANALYSIS")
    print("=" * 50)
    
    # Title statistics
    print(f"\n1. Title Statistics:")
    df['title_length'] = df['title'].str.len()
    print(f"   Average title length: {df['title_length'].mean():.1f} characters")
    print(f"   Shortest title: {df['title_length'].min()} characters")
    print(f"   Longest title: {df['title_length'].max()} characters")
    
    # Missing titles
    missing_titles = df['title'].isnull().sum()
    print(f"\n2. Missing titles: {missing_titles} ({missing_titles/len(df)*100:.1f}%)")
    
    # Common words in titles
    print(f"\n3. Common words in titles:")
    all_titles = ' '.join(df['title'].dropna().astype(str))
    words = all_titles.lower().split()
    word_counts = pd.Series(words).value_counts().head(10)
    print(word_counts)
    
    return df

def create_exploration_visualizations(df, year_counts, journal_counts):
    """
    Create basic visualizations for exploration.
    
    Args:
        df (pd.DataFrame): Dataset
        year_counts (pd.Series): Publication year counts
        journal_counts (pd.Series): Journal counts
    """
    print("\n" + "=" * 50)
    print("CREATING EXPLORATION VISUALIZATIONS")
    print("=" * 50)
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Publications over time
    plt.figure(figsize=(12, 6))
    year_counts.plot(kind='bar')
    plt.title('Number of Publications by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/publications_by_year.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Top journals
    plt.figure(figsize=(12, 8))
    journal_counts.head(15).plot(kind='barh')
    plt.title('Top 15 Journals by Publication Count', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Publications')
    plt.tight_layout()
    plt.savefig('plots/top_journals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Title length distribution
    plt.figure(figsize=(10, 6))
    df['title_length'].hist(bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Title Lengths', fontsize=14, fontweight='bold')
    plt.xlabel('Title Length (characters)')
    plt.ylabel('Frequency')
    plt.axvline(df['title_length'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["title_length"].mean():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/title_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Missing data heatmap
    plt.figure(figsize=(12, 8))
    missing_data = df.isnull()
    sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Pattern', fontsize=14, fontweight='bold')
    plt.xlabel('Columns')
    plt.tight_layout()
    plt.savefig('plots/missing_data_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to 'plots/' directory")

def main():
    """Main function to run data exploration."""
    print("CORD-19 Data Exploration")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Basic exploration
    missing_df = basic_exploration(df)
    
    # Publication date analysis
    year_counts = explore_publication_dates(df)
    
    # Journal analysis
    journal_counts = explore_journals(df)
    
    # Author analysis
    df_with_authors = explore_authors(df)
    
    # Title analysis
    df = explore_titles(df)
    
    # Create visualizations
    create_exploration_visualizations(df, year_counts, journal_counts)
    
    print("\n" + "=" * 50)
    print("EXPLORATION COMPLETE")
    print("=" * 50)
    print("Next steps:")
    print("1. Run data_analysis.py for detailed analysis")
    print("2. Run streamlit_app.py for interactive dashboard")

if __name__ == "__main__":
    main()
