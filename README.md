# Min Hashing
This python script was created for pre-processing and pairwise comparison (Exact jacquard similarity for baseline and MinHashing). The results were tested on comments scraped from:
- Amazon
- IMDB
- Yelp

## Folder contents
- FeatureVector_creation.py
- main.py

## Implementation
- Implemented in python 3.7
- The feature vector for the dataset along with the min hashing computation was creaeted from scratch
- Run code as: 
$ python main.py

## Output

For each k-value, the following output has been generated. Samples have been listed below:
- Baseline efficiency
- Min Hashing efficiency
- Time elapsed for signature generation
- Mean Squared error

### Sample output for k = 16:

Efficiency for baseline:  259.39309334754944
Efficiency for MinHashing:  42.278581857681274
Time taken to generate signatures:  6.789660215377808
Mean Squared Error:  4.934755947367183e-06

## Acknowledgements
The dataset was originally used in the following paper:

[1] From Group to Individual Labels using Deep Features", Kotzias et al., SIGKDD 2015
