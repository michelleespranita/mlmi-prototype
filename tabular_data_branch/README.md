### Get Started
There are 3 important notebooks:
1. Tabular Dataset Exploration.ipynb: Data pre-processing for the tabular data
2. MLP.ipynb: Uses pre-trained weights from CF_Mortality.model to build MLP to process the tabular data
3. FTTransformer.ipynb: Uses FT-Transformer to process the tabular data


### Datasets
1. CF_dataset(1).txt: Raw, unprocessed metadata (from Matthias, presumably from the original paper, however not on the website)
2. 220614_ictcf_data.csv: Processed metadata but unnormalized (from Matthias) -> 124 features
3. matt_metadata_unnorm.csv: Like 220614_ictcf_data.csv, but with additional columns (BCF8, PC, PS) -> 127 features
4. matt_metadata_norm.csv: Min-max normalized version of matt_metadata_unnorm.csv