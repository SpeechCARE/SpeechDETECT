# SpeechCARE: Speech Analysis Pipeline

## Overview
This is the repo of "SpeechCARE: An Explainable Automated Speech Processing Tool for Early Detection of Cognitive ImpAiRmEnt".
This study developed a comprehensive acoustic parameter set tailored to detect speech cues indicative of cognitive impairment. This set evaluates the vocal component across eight domains representing *vocal traits*—including Frequency Parameters, Cepstral Coefficients and Spectral Features, Voice Quality, Loudness, Intensity, and Speech Signal Complexity—and *speech temporal aspects*, including Speech Fluency, Rhythmic Structure, and Speech Production Dynamics. We assessed the effectiveness SpeechCARE using the DementiaBank dataset, a recognized benchmark that contains audio recordings from the "Cookie-Theft" picture-description task. Our evaluations, conducted with a machine learning classifier, revealed that Voice-Mark MCI offers significant insights into speech-related cognitive impairments, enhancing early detection strategies for patients at risk of ADRD.


## Dataset

We measured the performance of SpeechCARE using the DementiaBank speech corpus, which includes recordings from 237 subjects who participated in a picture description task. The subjects comprised 122 cognitively impaired and 115 cognitively normal individuals. The dataset was split into training and testing sets with the following characteristics:

#### Training Data

| Attributes                        | Case Group     | Control Group    |
|-----------------------------------|----------------|------------------|
| Participants                      | 87             | 79               |
| Gender (F/M)                      | 58 / 29        | 52 / 27          |
| Age (mean ± std)                  | 69.72 ± 6.80   | 66.04 ± 6.25     |
| MMSE score (mean ± std)           | 17.44 ± 5.33   | 28.99 ± 1.15     |

#### Testing Data

| Attributes                        | Case Group     | Control Group    |
|-----------------------------------|----------------|------------------|
| Participants                      | 35             | 36               |
| Gender (F/M)                      | 21 / 14        | 23 / 13          |
| Age (mean ± std)                  | 68.51 ± 7.12   | 66.11 ± 6.53     |
| MMSE score (mean ± std)           | 18.86 ± 5.80   | 28.91 ± 1.25     |


## Results
The performance of all machine learning classifiers, following hyperparameter optimization using 5-fold cross-validation, is reported in the following table, based on their evaluation on the test set:

ML classifier          |	F1-score      | 	AUC-ROC |
|----------------------|----------------|-----------|
|Random Forest          |	63.88        	|71.58      |
|Extra Trees	|60.52	|65.39 |
|AdaBoost	| 68.85 |	75.55 |
|XGBoost	| 60.60	| 62.22 |
|SVM	| 71.23	| 77.93 |
|**MLP**	| **80.5**	| **79.65** |]
