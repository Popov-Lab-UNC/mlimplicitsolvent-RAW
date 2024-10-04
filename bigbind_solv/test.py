from create_dataset import simulate_MAF_row
from create_dataset import simulate_row
import pandas as pd
import time 


start = time.time()
mock_data = {
    'bigbind_index': [4],  
    'lig_smiles': ['CN(C)C(=O)c1ccc(cc1)OC'], 
}
mock_df = pd.DataFrame(mock_data)
mock_row = mock_df.iloc[0]


nonMAF = time.time()

simulate_row(mock_row)

MAF = time.time()
print(f"MAF - {nonMAF - MAF}")
