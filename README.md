## How to Use (User Manual)

### Installation
pip install pandas numpy scikit-learn matplotlib seaborn

### Run Full Pipeline
python pipeline.py

### Run Visualization Only
python pipeline.py viz

## Function Reference

### run_data_science_pipeline(data_path, k_fold=5, save_plot=True)
- data_path : str  - Path to CSV file
- k_fold    : int  - Number of folds for cross-validation (default=5)
- save_plot : bool - Whether to save result figure (default=True)
- Returns   : dict - classification, regression, top5 results

### save_results(results, path='pipeline_results.pkl')
Saves pipeline results to pickle file for later reuse.

### load_and_visualize(pkl_path, save_path)
Loads saved results and regenerates visualization only.