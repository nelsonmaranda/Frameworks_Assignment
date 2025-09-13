"""
Sample Data Generator for CORD-19 Assignment
Creates a realistic sample dataset for testing and learning purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_cord19_data(n_samples=1000):
    """
    Generate a sample CORD-19 dataset for testing purposes.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample dataset with CORD-19-like structure
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Sample data for different fields
    journals = [
        'Nature', 'Science', 'The Lancet', 'NEJM', 'JAMA', 'BMJ',
        'PLOS ONE', 'Scientific Reports', 'Cell', 'Nature Medicine',
        'Lancet Infectious Diseases', 'Clinical Infectious Diseases',
        'Journal of Medical Virology', 'Virology', 'Emerging Infectious Diseases'
    ]
    
    authors = [
        'Smith, J.', 'Johnson, A.', 'Williams, B.', 'Brown, C.', 'Jones, D.',
        'Garcia, E.', 'Miller, F.', 'Davis, G.', 'Rodriguez, H.', 'Martinez, I.',
        'Hernandez, J.', 'Lopez, K.', 'Gonzalez, L.', 'Wilson, M.', 'Anderson, N.',
        'Thomas, O.', 'Taylor, P.', 'Moore, Q.', 'Jackson, R.', 'Martin, S.'
    ]
    
    # COVID-19 related keywords for titles
    covid_keywords = [
        'COVID-19', 'SARS-CoV-2', 'coronavirus', 'pandemic', 'vaccine',
        'transmission', 'mortality', 'hospitalization', 'lockdown', 'social distancing',
        'asymptomatic', 'variant', 'immunity', 'antibody', 'testing',
        'ventilator', 'ICU', 'elderly', 'comorbidity', 'outbreak'
    ]
    
    medical_terms = [
        'clinical trial', 'epidemiology', 'pathogenesis', 'treatment', 'diagnosis',
        'prognosis', 'risk factors', 'complications', 'recovery', 'long-term effects',
        'respiratory', 'cardiovascular', 'neurological', 'gastrointestinal', 'dermatological'
    ]
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        # Generate publication date (mostly 2020-2022)
        start_date = datetime(2019, 12, 1)
        end_date = datetime(2022, 12, 31)
        random_days = random.randint(0, (end_date - start_date).days)
        publish_time = start_date + timedelta(days=random_days)
        
        # Generate title
        title_keywords = random.sample(covid_keywords, random.randint(1, 3))
        medical_term = random.choice(medical_terms)
        title = f"{' '.join(title_keywords).title()}: A {medical_term} Study"
        
        # Generate abstract (simplified)
        abstract = f"This study investigates {random.choice(covid_keywords)} in relation to {medical_term}. " \
                  f"Our findings suggest significant implications for {random.choice(['treatment', 'prevention', 'diagnosis'])} " \
                  f"of COVID-19. The research was conducted using {random.choice(['retrospective', 'prospective', 'cross-sectional'])} " \
                  f"analysis of {random.randint(50, 5000)} patients."
        
        # Generate authors (1-5 authors per paper)
        n_authors = random.randint(1, 5)
        paper_authors = random.sample(authors, n_authors)
        authors_str = '; '.join(paper_authors)
        
        # Generate source
        source = random.choice(['PubMed', 'PMC', 'WHO', 'CDC', 'Custom'])
        
        # Generate some missing values (realistic for real datasets)
        has_abstract = random.random() > 0.1  # 90% have abstracts
        has_authors = random.random() > 0.05  # 95% have authors
        
        data.append({
            'cord_uid': f'cord_uid_{i:06d}',
            'sha': f'sha_{i:08x}' if random.random() > 0.3 else None,
            'source_x': source,
            'title': title,
            'doi': f'10.1000/{i:06d}' if random.random() > 0.2 else None,
            'pmcid': f'PMC{i:07d}' if random.random() > 0.4 else None,
            'pubmed_id': f'{random.randint(100000, 999999)}' if random.random() > 0.3 else None,
            'license': random.choice(['cc0', 'cc-by', 'cc-by-nc', 'custom']),
            'abstract': abstract if has_abstract else None,
            'publish_time': publish_time.strftime('%Y-%m-%d'),
            'authors': authors_str if has_authors else None,
            'journal': random.choice(journals),
            'mag_id': f'mag_{i:08d}' if random.random() > 0.5 else None,
            'who_covidence_id': f'cov_{i:06d}' if random.random() > 0.7 else None,
            'arxiv_id': f'arXiv:{random.randint(2000, 2022)}.{random.randint(1000, 9999)}' if random.random() > 0.8 else None,
            'pdf_json_files': f'pdf_{i:06d}.json' if random.random() > 0.6 else None,
            'pmc_json_files': f'pmc_{i:06d}.json' if random.random() > 0.7 else None,
            'url': f'https://example.com/paper/{i}' if random.random() > 0.4 else None,
            's2_id': f's2_{i:08d}' if random.random() > 0.6 else None
        })
    
    return pd.DataFrame(data)

def main():
    """Generate and save sample data."""
    print("Generating sample CORD-19 dataset...")
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data
    df = generate_sample_cord19_data(n_samples=2000)
    
    # Save to CSV
    output_path = 'data/sample_metadata.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset generated with {len(df)} rows")
    print(f"Saved to: {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Display basic info
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
