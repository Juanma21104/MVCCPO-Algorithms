import pandas as pd

# This module contains functions to load datasets for portfolio optimization
def load_dataset(file_path):
    """Load dataset and return expected returns and covariance matrix using only pandas."""

    # Read the entire file as lines using pandas
    raw_lines = pd.read_csv(file_path, header=None, dtype=str).squeeze().tolist()
    
    # Get number of assets
    num_assets = int(raw_lines[0])

    # Parse expected returns and std deviations
    stats_df = pd.read_csv(
        file_path, 
        sep=r'\s+', 
        skiprows=1, 
        nrows=num_assets, 
        header=None, 
        names=['return', 'std_dev']
    ).astype(float)

    # Initialize correlation matrix
    corr_matrix = pd.DataFrame(1.0, index=range(num_assets), columns=range(num_assets))

    # Read correlation data from the file
    correlations = pd.read_csv(
        file_path,
        sep=r'\s+',
        skiprows=num_assets + 1,
        header=None,
        names=['i', 'j', 'corr']
    )

    # Fill correlation matrix
    for _, row in correlations.iterrows():
        i_idx, j_idx = int(row['i']) - 1, int(row['j']) - 1
        corr = float(row['corr'])
        corr_matrix.iat[i_idx, j_idx] = corr
        corr_matrix.iat[j_idx, i_idx] = corr

    # Compute covariance matrix
    std_devs = stats_df['std_dev']
    std_outer = std_devs.to_frame().dot(std_devs.to_frame().T)
    cov_matrix = std_outer * corr_matrix

    return stats_df['return'].values, cov_matrix.values
