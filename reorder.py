import sys
import pandas as pd

target = pd.read_csv(sys.argv[1])
reorder = pd.read_csv(sys.argv[2])

reorder = reorder[target.keys()]
reorder = reorder.rename({"Unnamed: 0": ""}, axis='columns')
reorder.to_csv(sys.argv[2], index=False)
