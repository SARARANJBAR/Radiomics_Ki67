# Radiomics ML

## Context

This radiomics experiment is focused on connecting tissue properties with imaging patterns. The data originated from two MRI-localized biopsy cohorts of GBM patients who were conscented and enrolled in institutions biopsy collection programs where multiple image localized biopsies were extracted from their tumor prior to gross or subtotal resection. Samples were sent to pathology for tissue analysis and RNAseq or Immunohistochemistry.

The aim of the project is to build a machine learning model to predict the abbundance of the marker in biopsies based on different imaging features describing patterns around the biopsies. We aim to minimise the difference between the real and the estimated abbundance of the target at each biopsy location. Model performance was evaluated using metrics such as mean squared error (mse), root squared of the mean squared error (rmse), and r-squared (r2).


## Feature generation

We had coregistered imaging data associated with the same time point as tissue extraction, including qualitative MRI sequences such as T1-weighted post contrast injection (T1Gd), T2-weighted (T2), and a quantitative MRI, apparent diffusion coefficients (ADC). 

Each of these image type were preprocessed offline, appropriate for the type of MRI that they belonged to, and quantitative features were extracted from a small area around each biopsy location using the pyradiomics pipeline. The result is a input csv file where each row data related to a unique biopsy, including  pyradiomics imaging features from 3 MRI types as well as the target, and some potentially biologically relevant features such as patient sex, age at death if available, type of tumor: recurrent or primary, and the source institution.

## Target

The target marker in this project was ki67, associated with proliferation is biospy samples. We know that proliferation is elevated where tumor cells are present. If we can predict where proliferation happens we can basically create maps corresponding to the spatial distribution of tumor cells across whole tumors. So basically identifying where tumor cells are. These maps can theoretically inform radiation plans, improving the efficacy of radiation therapy.


### How do I download the dataset?

Proprietary data, sorry cant share it.
