import splitfolders

# Split waste_dataset into 80% train and 20% test
splitfolders.ratio("waste_dataset", output="data_split", seed=42, ratio=(.8, .2), move=False)

