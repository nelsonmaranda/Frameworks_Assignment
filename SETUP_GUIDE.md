# CORD-19 Assignment Setup Guide

## Quick Start

### 1. Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### 2. Installation
```bash
# Navigate to the assignment directory
cd "C:\PLP\Python\Week 8 assignment"

# Install required packages
pip install -r requirements.txt
```

### 3. Generate Sample Data
```bash
# Generate sample dataset (if you don't have the real CORD-19 data)
python sample_data_generator.py
```

### 4. Run the Analysis
```bash
# Step 1: Data exploration
python data_exploration.py

# Step 2: Data analysis and visualization
python data_analysis.py

# Step 3: Launch interactive dashboard
streamlit run streamlit_app.py
```

## Detailed Instructions

### Part 1: Data Loading and Exploration
The `data_exploration.py` script performs:
- ✅ Data loading and basic information display
- ✅ Missing value analysis
- ✅ Publication date analysis
- ✅ Journal and author analysis
- ✅ Title content analysis
- ✅ Basic visualizations

**Output:** Creates plots in the `plots/` directory

### Part 2: Data Analysis and Visualization
The `data_analysis.py` script performs:
- ✅ Publication trend analysis
- ✅ Journal concentration analysis
- ✅ Title and abstract content analysis
- ✅ Word frequency analysis
- ✅ Advanced visualizations
- ✅ Comprehensive reporting

**Output:** Creates detailed plots and generates analysis report

### Part 3: Interactive Streamlit Application
The `streamlit_app.py` provides:
- ✅ Interactive data filtering
- ✅ Real-time visualizations
- ✅ Search functionality
- ✅ Data exploration tools
- ✅ Export capabilities

**Usage:** Open your browser to the URL shown in the terminal (usually http://localhost:8501)

## File Structure

```
Week 8 assignment/
├── requirements.txt              # Python dependencies
├── README.md                    # Project overview
├── SETUP_GUIDE.md              # This file
├── sample_data_generator.py    # Generate sample data
├── data_exploration.py         # Part 1: Data exploration
├── data_analysis.py            # Part 2: Analysis & visualization
├── streamlit_app.py            # Part 3: Interactive dashboard
├── data/                       # Data directory
│   └── sample_metadata.csv     # Sample dataset
└── plots/                      # Generated visualizations
    ├── publications_by_year.png
    ├── top_journals.png
    ├── title_length_distribution.png
    └── ... (more plots)
```

## Features Implemented

### ✅ Data Exploration
- Dataset dimensions and structure
- Missing value analysis
- Publication date patterns
- Journal and author statistics
- Title content analysis

### ✅ Data Analysis
- Publication trends over time
- Journal concentration metrics
- Word frequency analysis
- Content length distributions
- Growth rate calculations

### ✅ Visualizations
- Publication timeline charts
- Journal distribution plots
- Word clouds
- Histograms and bar charts
- Interactive Plotly charts

### ✅ Streamlit Dashboard
- Interactive filtering
- Real-time data exploration
- Search functionality
- Export capabilities
- Responsive design

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **File not found**: Run `python sample_data_generator.py` first
3. **Streamlit not starting**: Check if port 8501 is available
4. **Memory issues**: Reduce sample size in `sample_data_generator.py`

### Performance Tips

- For large datasets, consider using a subset
- Close other applications to free up memory
- Use the sample data generator for testing

## Assignment Requirements Checklist

- ✅ **Data Loading**: Load and examine dataset structure
- ✅ **Data Cleaning**: Handle missing values and data types
- ✅ **Basic Analysis**: Count papers by year, top journals, word frequency
- ✅ **Visualizations**: Publication trends, journal charts, word clouds
- ✅ **Streamlit App**: Interactive dashboard with filtering
- ✅ **Documentation**: Comprehensive README and setup guide
- ✅ **Code Quality**: Well-commented, readable code
- ✅ **GitHub Ready**: All files organized for repository upload

## Next Steps

1. **Test all components** to ensure they work correctly
2. **Customize visualizations** if needed
3. **Add real CORD-19 data** by downloading from Kaggle
4. **Create GitHub repository** and upload files
5. **Submit repository URL** for assignment completion

## Support

If you encounter any issues:
1. Check the error messages carefully
2. Ensure all dependencies are installed
3. Verify file paths are correct
4. Check Python version compatibility

The project demonstrates proficiency in:
- Data loading and exploration
- Data cleaning and preparation
- Statistical analysis
- Visualization creation
- Interactive web application development
- Documentation and presentation
