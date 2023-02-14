# Towards Feature Validation in DNNs
This repository contains the code to reproduce the results found in the 
paper "Towards Feature Validation in Time to Lane Change Classification 
using Deep Neural Networks" by O. De Candido, M. Koller, O. Gallitz, R. Melz,
M. Botsch, and W. Utschick, published at the [2020 IEEE ITSC](https://ieeexplore.ieee.org/document/9294555) conference.

## How to use the code
1. Store the training data in `data/` folder, e.g., the extracted lane 
   changes from the highD 
   dataset from this [project](https://github.com/decandido/highD-extract-lane-changes).
2. Run `main.py` to train the DNNs with the network architectures stored in 
   `parameters/params.py`.
3. Run `extract_embeddings.py` to extract the feature embeddings from the 
   trained networks, and calculate k-means on those embeddings.
4. Run `umap_embeddings.py` to calculate the [UMAP](https://umap-learn.readthedocs.io/en/latest/) representations of the feature embeddings.
5. (Optional) Update the network architectures in `parameters/params.py` and 
   rerun the scripts.

## Requirements
The required packages can be installed via the `requirements.txt` file, e.g.,
`conda install -r requirements.txt`.
This code was tested using `Python v3.7.0`.

## Paper Reference
```angular2html
@inproceedings{decandido2020towards,
        author={De Candido, Oliver and Koller, Michael and Gallitz, Oliver and Melz, Ron and Botsch, Michael and Utschick, Wolfgang},
        booktitle={Proc. IEEE 23rd Intell. Transp. Syst. Conf. (ITSC)},
        title={Towards Feature Validation in Time to Lane Change Classification using Deep Neural Networks}, 
        year={2020},
        pages={1697--1704},
        publisher={IEEE},
        doi={10.1109/ITSC45102.2020.9294555}
}
```