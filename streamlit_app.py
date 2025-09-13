"""
Streamlit Application for CORD-19 Data Explorer
Part 4: Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    # Try multiple possible data locations
    data_paths = [
        'data/sample_metadata.csv',
        'sample_metadata.csv',
        './data/sample_metadata.csv'
    ]
    
    for path in data_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Convert publish_time to datetime
                df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
                df['year'] = df['publish_time'].dt.year
                df['month'] = df['publish_time'].dt.month
                df['abstract_word_count'] = df['abstract'].str.split().str.len()
                df['title_word_count'] = df['title'].str.split().str.len()
                return df
        except Exception as e:
            continue
    
    # If no data file found, show error with instructions
    st.error("""
    **Data file not found!** 
    
    For local development:
    1. Run `python sample_data_generator.py` to create sample data
    
    For deployment:
    - Make sure the data file is included in your repository
    - The app will automatically find the data file
    """)
    return None

def create_word_cloud(text_data, title="Word Cloud"):
    """Create a word cloud visualization."""
    if text_data is None or text_data.empty or text_data.isnull().all():
        return None
    
    # Combine all text
    all_text = ' '.join(text_data.dropna().astype(str))
    
    if not all_text.strip():
        return None
    
    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
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
    
    if not filtered_words:
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap='viridis',
        relative_scaling=0.5
    ).generate(' '.join(filtered_words))
    
    return wordcloud

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Simple exploration of COVID-19 research papers")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Year range filter
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider(
        "Select year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    # Journal filter
    all_journals = ['All'] + sorted(df['journal'].dropna().unique().tolist())
    selected_journal = st.sidebar.selectbox(
        "Select journal",
        all_journals
    )
    
    # Source filter
    all_sources = ['All'] + sorted(df['source_x'].dropna().unique().tolist())
    selected_source = st.sidebar.selectbox(
        "Select source",
        all_sources
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Year filter
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]
    
    # Journal filter
    if selected_journal != 'All':
        filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
    
    # Source filter
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source_x'] == selected_source]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Papers",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        unique_journals = filtered_df['journal'].nunique()
        st.metric(
            label="Unique Journals",
            value=f"{unique_journals:,}"
        )
    
    with col3:
        papers_with_abstracts = filtered_df['abstract'].notna().sum()
        st.metric(
            label="Papers with Abstracts",
            value=f"{papers_with_abstracts:,}",
            delta=f"{(papers_with_abstracts/len(filtered_df)*100):.1f}%" if len(filtered_df) > 0 else "0%"
        )
    
    with col4:
        avg_title_length = filtered_df['title'].str.len().mean()
        st.metric(
            label="Avg Title Length",
            value=f"{avg_title_length:.0f} chars"
        )
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Trends", "📚 Journals", "📝 Content", "🔍 Data Explorer"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Publication timeline
            st.subheader("Publications by Year")
            yearly_counts = filtered_df['year'].value_counts().sort_index()
            
            fig = px.bar(
                x=yearly_counts.index,
                y=yearly_counts.values,
                title="Number of Publications by Year",
                labels={'x': 'Year', 'y': 'Number of Publications'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Source distribution
            st.subheader("Publications by Source")
            source_counts = filtered_df['source_x'].value_counts()
            
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Distribution by Source"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing data heatmap
        st.subheader("Missing Data Pattern")
        missing_data = filtered_df.isnull()
        
        # Create a smaller sample for the heatmap if dataset is large
        sample_size = min(1000, len(missing_data))
        sample_missing = missing_data.sample(n=sample_size, random_state=42)
        
        fig = px.imshow(
            sample_missing.T,
            title="Missing Data Pattern (Sample)",
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Publication Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly trends
            st.subheader("Monthly Publication Trends")
            filtered_df['year_month'] = filtered_df['publish_time'].dt.to_period('M')
            monthly_counts = filtered_df['year_month'].value_counts().sort_index()
            
            # Convert period index to string for plotting
            monthly_counts.index = monthly_counts.index.astype(str)
            
            fig = px.line(
                x=monthly_counts.index,
                y=monthly_counts.values,
                title="Monthly Publication Counts",
                labels={'x': 'Month', 'y': 'Number of Publications'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Growth rate
            st.subheader("Year-over-Year Growth")
            yearly_counts = filtered_df['year'].value_counts().sort_index()
            growth_rates = yearly_counts.pct_change() * 100
            
            fig = px.bar(
                x=growth_rates.index[1:],
                y=growth_rates.values[1:],
                title="Year-over-Year Growth Rate (%)",
                labels={'x': 'Year', 'y': 'Growth Rate (%)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Journal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top journals
            st.subheader("Top Publishing Journals")
            journal_counts = filtered_df['journal'].value_counts().head(15)
            
            fig = px.bar(
                x=journal_counts.values,
                y=journal_counts.index,
                orientation='h',
                title="Top 15 Journals by Publication Count",
                labels={'x': 'Number of Publications', 'y': 'Journal'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Journal concentration
            st.subheader("Journal Concentration")
            journal_counts = filtered_df['journal'].value_counts()
            cumulative_pct = (journal_counts.cumsum() / journal_counts.sum()) * 100
            
            fig = px.line(
                x=range(1, len(cumulative_pct) + 1),
                y=cumulative_pct,
                title="Cumulative Share of Publications",
                labels={'x': 'Number of Journals (ranked)', 'y': 'Cumulative %'}
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text="80% threshold")
            fig.add_hline(y=90, line_dash="dash", line_color="orange", 
                         annotation_text="90% threshold")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Title word cloud
            st.subheader("Most Common Words in Titles")
            wordcloud = create_word_cloud(filtered_df['title'], "Title Word Cloud")
            
            if wordcloud:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud: Most Frequent Words in Titles')
                st.pyplot(fig)
            else:
                st.info("No title data available for word cloud generation")
        
        with col2:
            # Title length distribution
            st.subheader("Title Length Distribution")
            title_lengths = filtered_df['title'].str.len()
            
            fig = px.histogram(
                x=title_lengths,
                nbins=30,
                title="Distribution of Title Lengths",
                labels={'x': 'Title Length (characters)', 'y': 'Frequency'}
            )
            fig.add_vline(x=title_lengths.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {title_lengths.mean():.1f}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Abstract analysis (if available)
        if filtered_df['abstract'].notna().sum() > 0:
            st.subheader("Abstract Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Abstract word cloud
                st.write("**Most Common Words in Abstracts**")
                abstract_wordcloud = create_word_cloud(filtered_df['abstract'], "Abstract Word Cloud")
                
                if abstract_wordcloud:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(abstract_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Word Cloud: Most Frequent Words in Abstracts')
                    st.pyplot(fig)
            
            with col2:
                # Abstract length distribution
                abstract_lengths = filtered_df['abstract'].str.len()
                
                fig = px.histogram(
                    x=abstract_lengths,
                    nbins=30,
                    title="Distribution of Abstract Lengths",
                    labels={'x': 'Abstract Length (characters)', 'y': 'Frequency'}
                )
                fig.add_vline(x=abstract_lengths.mean(), line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {abstract_lengths.mean():.0f}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Data Explorer")
        
        # Search functionality
        st.subheader("Search Papers")
        search_term = st.text_input("Search in titles and abstracts:", "")
        
        if search_term:
            # Search in titles and abstracts
            title_matches = filtered_df['title'].str.contains(search_term, case=False, na=False)
            abstract_matches = filtered_df['abstract'].str.contains(search_term, case=False, na=False)
            search_results = filtered_df[title_matches | abstract_matches]
            
            st.write(f"Found {len(search_results)} papers matching '{search_term}'")
            
            if len(search_results) > 0:
                # Display search results
                for idx, row in search_results.head(10).iterrows():
                    with st.expander(f"**{row['title']}** ({row['year']})"):
                        st.write(f"**Journal:** {row['journal']}")
                        st.write(f"**Authors:** {row['authors']}")
                        st.write(f"**Abstract:** {row['abstract'][:500]}..." if pd.notna(row['abstract']) else "No abstract available")
                        if pd.notna(row['url']):
                            st.write(f"**URL:** {row['url']}")
        
        # Data table
        st.subheader("Data Table")
        
        # Select columns to display
        available_columns = ['title', 'authors', 'journal', 'year', 'source_x', 'abstract']
        selected_columns = st.multiselect(
            "Select columns to display:",
            available_columns,
            default=['title', 'authors', 'journal', 'year']
        )
        
        if selected_columns:
            display_df = filtered_df[selected_columns].head(100)
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"cord19_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**CORD-19 Data Explorer** | Built with Streamlit | "
        "Data source: CORD-19 Research Challenge"
    )

if __name__ == "__main__":
    main()
