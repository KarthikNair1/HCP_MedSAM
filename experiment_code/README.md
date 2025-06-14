# Detailed Methods

## MedSAM Training:

During training, we explored and determined the following parameters using performance on the validation set:

Starting from a SAM checkpoint versus a MedSAM checkpoint
- Learning rate ∈ [.00005 .0001 **.0005** .001 .0025 .005 .01] 
- The percent of slices without the class of interest to keep in the training set ∈ [0 0.1% 1% 5% **10%** 25% 100%]
- Weighting between cross-entropy loss and DICE loss (**100% DICE**)
- Number of epochs (**10**)

Models were trained with a batch size of 256 using AdamW optimization with a weight decay of .01. 
- Three regions (ctx-lh-entorhinal, ctx-rh-entorhinal, and Right-Accumbens-area) failed to converge properly (defined as having a maximum validation accuracy ≤ .01) when trained using MedSAM-constant. '
- These regions were subsequently retrained with a learning rate of .0001, which produced convergence.

## UNet Training

During training, we explored and determined the following parameters using performance on the validation set:

- Learning rate ∈ [.00005 .0001 **.0005** .001]
- The percent of slices without the class of interest to keep in the training set ∈ [0% .1% 1% 5% **10%** 25% 100%]
- Weighting between cross-entropy loss and DICE loss (**100% DICE**)
- Number of epochs (**10**)

Models were trained using a batch size of 64 with Adam optimization. 

## Dataset Ablation Analysis

Training data were subsetted to vary the number of imaging volumes used in training from 1 to 891, providing a spectrum of dataset sizes (i.e.,1, 2, 3, 4, 5, 7, 10, 15, 20, 50, 100, 250, 500, and 891 imaging volumes). MedSAM and UNet models were trained using each subset size for 2 hours. 

The model performance was dependent on batch size under this fixed time budget, so the ideal batch size was determined for each (model, label) pair using validation data. Models were evaluated against the same validation or test set to ensure consistent comparison. 

Model training was repeated 5 times for each subset size using different samples of the data to estimate variability in performance. Repeats with a validation Dice score less than .01 were excluded from downstream analysis due to being stuck in the local minimum of predicting the background class. This resulted in 3% of repeats removed from the left caudate MedSAM models, and 10% of repeats removed from the left insula MedSAM models.

