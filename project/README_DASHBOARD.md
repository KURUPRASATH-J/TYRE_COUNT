# Height Measurement Dashboard

A comprehensive Streamlit dashboard for visualizing stereo vision height measurement results.

## Features

- 📊 **Real-time Statistics**: View median, mean, min/max heights, and standard deviation
- 📈 **Interactive Charts**: 
  - Height measurements over time
  - Height distribution histogram
  - Camera confidence scores
  - Detection box areas
  - Height vs confidence correlation
- 🔍 **Quality Metrics**: Track detection success rates and quality rejections
- 📋 **Data Table**: Browse all measurements with filtering options
- 📥 **Data Export**: Download measurements as CSV

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. **Run the height measurement script** (if you haven't already):
```bash
python height.py
```

This will generate `measurement_data.json` with all measurement data.

2. **Launch the dashboard**:
```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your web browser at `http://localhost:8501`

## Dashboard Sections

### Key Metrics
- Median Height: The median of all valid measurements
- Mean Height: Average height with standard deviation
- Height Range: Min and max values
- Standard Deviation: Measurement variability

### Visualizations
- **Height Over Time**: Line chart showing height measurements across frames
- **Height Distribution**: Histogram with mean and median indicators
- **Confidence Scores**: Track camera confidence over time
- **Box Areas**: Monitor detection box sizes
- **Correlation Analysis**: Height vs confidence scatter plots

### Quality Metrics
- Frames processed
- Successful detections
- Quality rejections
- Success rate percentage

### Data Table
- View all measurements in a sortable, filterable table
- Export data as CSV

## Notes

- The dashboard automatically loads data from `measurement_data.json`
- If no data is found, the dashboard will show a warning message
- Use the sidebar filters to adjust confidence thresholds
- All charts are interactive - hover for details, zoom, and pan

## Troubleshooting

**No data found:**
- Make sure you've run `height.py` first
- Check that `measurement_data.json` exists in the same directory

**Dashboard not loading:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that Streamlit is installed: `pip install streamlit`

