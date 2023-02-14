# Towards Feature Validation in DNNs
This repository contains the code to reproduce the results found in the 
paper "Towards Feature Validation in Time to Lane Change Classification 
using Deep Neural Networks" by O. De Candido, M. Koller, O. Gallitz, R. Melz,
M. Botsch, and W. Utschick, published at the 2020 IEEE ITSC conference.

### Paper Abstract
In this paper, we explore different Convolutional Neural Network (CNN) architectures to extract features in a Time to Lane Change (TTLC) classification problem for highway driving functions. These networks are trained using the HighD dataset, a public dataset of realistic driving on German highways. The investigated CNNs achieve approximately the same test accuracy which, at first glance, seems to suggest that all of the algorithms extract features of equal quality. We argue however that the test accuracy alone is not sufficient to validate the features which the algorithms extract. As a form of validation, we propose a two pronged approach to confirm the quality of the extracted features. In the first stage, we apply a clustering algorithm on the features and investigate how logical the feature clusters are with respect to both an external clustering validation measure and with respect to expert knowledge. In the second stage, we use a state-of-the-art dimensionality reduction technique to visually support the findings of the first stage of validation. In the end, our analysis suggests that the different CNNs, which have approximately equal accuracies, extract features of different quality. This may lead a user to choose one of the CNN architectures over the others.

### Paper Reference
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
