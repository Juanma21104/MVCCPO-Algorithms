import numpy as np

def load_dataset(file_path):
    """Load the dataset from a file and return expected returns and covariance matrix."""

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # The first line contains the number of assets
    num_assets = int(lines[0].strip())
    returns = []
    std_devs = []
    
    # Read the expected returns and standard deviations
    for i in range(1, num_assets + 1):
        data = list(map(float, lines[i].strip().split()))
        returns.append(data[0])  # Expected return
        std_devs.append(data[1]) # Standard deviation

    # Convert to numpy arrays
    returns = np.array(returns)
    std_devs = np.array(std_devs)
    
    corr_matrix = np.eye(num_assets)  # Initialize with identity matrix

    # Read the correlation coefficients
    print("numero de lineas: ", len(lines), " --- num_assets: ", num_assets)
    for i in range(num_assets + 1, len(lines)):
        if(lines[i].strip().split()) == []:
            continue
        i_idx, j_idx, corr = lines[i].strip().split()
        i_idx, j_idx = int(i_idx) - 1, int(j_idx) - 1 # Convert to 0-based index
        corr = float(corr)
        corr_matrix[i_idx, j_idx] = corr
        corr_matrix[j_idx, i_idx] = corr  # Symmetric matrix
    
    # Calculate the covariance matrix
    # print("Standard Deviations:", std_devs)
    # print("Correlation Matrix:\n", corr_matrix)
    cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
    
    return returns, cov_matrix