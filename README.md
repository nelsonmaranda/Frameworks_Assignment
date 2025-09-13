# Week 8 Assignment: CORD-19 Data Analysis

## 📋 Assignment Overview
This assignment involves analyzing the CORD-19 research dataset and creating a Streamlit application to display findings. The project focuses on fundamental data analysis skills including data loading, cleaning, visualization, and interactive web application development.

## 🎯 Learning Objectives
- Practice loading and exploring real-world datasets
- Learn basic data cleaning techniques
- Create meaningful visualizations
- Build a simple interactive web application
- Present data insights effectively

## 📊 Dataset Information
We work with the metadata.csv file from the CORD-19 dataset, which contains information about COVID-19 research papers including:
- Paper titles and abstracts
- Publication dates
- Authors and journals
- Source information

**Note:** For this assignment, we include a sample data generator to create realistic test data if the full CORD-19 dataset is not available.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation & Setup
```bash
# 1. Navigate to the assignment directory
cd "C:\PLP\Python\Week 8 assignment"

# 2. Install required packages
pip install -r requirements.txt

# 3. Generate sample data (if needed)
python sample_data_generator.py

# 4. Run the complete analysis
python data_exploration.py    # Part 1: Data exploration
python data_analysis.py       # Part 2: Analysis & visualization
streamlit run streamlit_app.py # Part 3: Interactive dashboard
```

## 📁 Project Structure
```
Week 8 assignment/
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── SETUP_GUIDE.md              # Detailed setup instructions
├── sample_data_generator.py    # Generate sample data for testing
├── data_exploration.py         # Part 1: Data loading and exploration
├── data_analysis.py            # Part 2: Analysis and visualization
├── streamlit_app.py            # Part 3: Streamlit web application
├── data/                       # Data directory
│   └── sample_metadata.csv     # Sample dataset (generated)
└── plots/                      # Generated visualizations
    ├── publications_by_year.png
    ├── top_journals.png
    ├── title_length_distribution.png
    └── ... (more plots)
```

## 🔧 Required Tools
- **pandas** (data manipulation)
- **matplotlib/seaborn** (visualization)
- **streamlit** (web application)
- **wordcloud** (word cloud generation)
- **plotly** (interactive charts)

## 📈 Features Implemented

### ✅ Part 1: Data Loading and Basic Exploration
- Dataset dimensions and structure analysis
- Missing value identification and analysis
- Publication date pattern exploration
- Journal and author statistics
- Title content analysis
- Basic visualizations

### ✅ Part 2: Data Cleaning and Preparation
- Handle missing data appropriately
- Convert date columns to datetime format
- Extract year from publication date
- Create new analytical columns (word counts, etc.)
- Data quality assessment

### ✅ Part 3: Data Analysis and Visualization
- Publication trends over time
- Journal concentration analysis
- Word frequency analysis in titles and abstracts
- Content length distributions
- Growth rate calculations
- Advanced statistical analysis

### ✅ Part 4: Streamlit Application
- Interactive data filtering by year, journal, and source
- Real-time visualization updates
- Search functionality across titles and abstracts
- Data export capabilities
- Responsive dashboard design
- Multiple analysis tabs

### ✅ Part 5: Documentation and Reflection
- Comprehensive code documentation
- Detailed analysis reports
- Setup and usage guides
- Performance insights and findings

## 📊 Key Insights Generated

### Publication Trends
- **Total publications analyzed:** 2,000 (sample dataset)
- **Date range:** 2019-12-01 to 2022-12-31
- **Peak year:** 2022 with 680 publications
- **Growth patterns:** Significant increase during pandemic years

### Journal Analysis
- **Total journals:** 15 unique journals
- **Top 10 journals account for:** 69.7% of publications
- **Gini coefficient (inequality):** 0.043 (relatively equal distribution)
- **Most prolific journal:** Science (150 papers)

### Content Analysis
- **Average title length:** 41.7 characters
- **Papers with abstracts:** 1,800 (90%)
- **Average abstract length:** 210 characters
- **Most common word:** 'study' (appears in all titles)

## 🎨 Visualizations Created

1. **Publication Timeline Charts**
   - Yearly publication counts
   - Monthly trends
   - Growth rate analysis

2. **Journal Analysis Charts**
   - Top publishing journals
   - Journal concentration curves
   - Distribution analysis

3. **Content Analysis Visualizations**
   - Word clouds for titles and abstracts
   - Title length distributions
   - Abstract length analysis

4. **Interactive Dashboard**
   - Real-time filtering
   - Search functionality
   - Export capabilities

## 🚀 Usage Instructions

### For Data Exploration
```bash
python data_exploration.py
```
**Output:** Basic dataset analysis and initial visualizations

### For Detailed Analysis
```bash
python data_analysis.py
```
**Output:** Comprehensive analysis report and advanced visualizations

### For Interactive Dashboard
```bash
streamlit run streamlit_app.py
```
**Output:** Web application accessible at http://localhost:8501

## 🔍 Advanced Features

### Interactive Filtering
- Filter by publication year range
- Select specific journals
- Choose data sources
- Real-time updates

### Search Capabilities
- Search across titles and abstracts
- Case-insensitive matching
- Results highlighting
- Export search results

### Data Export
- CSV export functionality
- Filtered data download
- Custom column selection
- Timestamped filenames

## 📚 Learning Outcomes

By completing this assignment, students will have:

1. **Data Science Workflow Experience**
   - Data loading and exploration
   - Data cleaning and preparation
   - Statistical analysis
   - Visualization creation

2. **Technical Skills**
   - Python programming with pandas
   - Data visualization with matplotlib/seaborn
   - Interactive web development with Streamlit
   - Data analysis best practices

3. **Presentation Skills**
   - Creating compelling visualizations
   - Building interactive dashboards
   - Documenting analysis findings
   - Presenting data insights effectively

## 🛠️ Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **File not found**: Run `python sample_data_generator.py` first
3. **Streamlit not starting**: Check if port 8501 is available
4. **Memory issues**: Reduce sample size in data generator

### Performance Tips
- Use sample data for testing
- Close other applications to free memory
- Consider data subset for large datasets

## 📝 Next Steps

1. **Test all components** to ensure functionality
2. **Customize visualizations** as needed
3. **Add real CORD-19 data** from Kaggle
4. **Create GitHub repository** and upload files
5. **Submit repository URL** for assignment completion

## 🤝 Support

For technical issues:
1. Check error messages carefully
2. Verify all dependencies are installed
3. Ensure file paths are correct
4. Check Python version compatibility

---

## 📦 Repository Information

**Repository Name:** `Frameworks_Assignment`  
**Assignment:** Week 8 - CORD-19 Data Analysis  
**Course:** Python Programming  
**Student:** [Nelson Maranda]  
**Date:** January 2025
