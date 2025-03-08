# RespiraCheck

## Overview
RespiraCheck is a convolutional neural network (CNN) designed to classify COVID-19 based on cough audio. Our approach utilizes Mel spectrogram representations of labeled cough recordings to fine-tune the fully connected layer of a pretrained ResNet-18 model, leveraging transfer learning for efficient and accurate classification. 

Using the COUGHVID dataset, we train on a balanced set of COVID-19 positive and negative samples, applying data augmentation techniques such as time and frequency masking to allow for flexibility in use, accommodating a wide variety of crowdsourced samples. To enhance real-world applicability, we have also developed a user-friendly interface that enables individuals to record or upload cough samples and receive an instant diagnostic assessment. By bridging the gap between clinical research and practical deployment, RespiraCheck aims to provide an accessible, non-invasive, and scalable tool for COVID-19 screening.

## Dataset

### Name: [COUGHVID](https://zenodo.org/record/4048312)
**Description:** The COUGHVID dataset is a collection of cough recordings gathered via a mobile app to assist in diagnosing COVID-19. It includes labeled audio samples from users with varying health conditions, including COVID-19-positive, healthy, and other respiratory illnesses.  
**Size:** Over 27,000 cough recordings.  
**Preprocessing:**  
- Noise reduction was applied to enhance signal clarity.  
- Spectrograms were generated with a consistent time-frequency resolution.  
- Normalized features to standardize amplitude and frequency distributions.  

### Name: [COSWARA](https://github.com/iiscleap/Coswara-Data)
**Description:** The COSWARA dataset contains respiratory audio recordings, including coughs, breathing sounds, and speech, collected to aid in COVID-19 detection. Participants self-reported their health status, allowing for labeled data across COVID-positive, asymptomatic, and healthy individuals.  
**Size:** Over 2,000 recordings from 1,500+ participants.  
**Preprocessing:** Same as above.  

## Model Architecture
RespiraCheck uses transfer learning to maximize prediction accuracy on limited amounts of training data. Both ResNet-18 and EfficientNet were used in our research, chosen for their lightweight architecture and the diversity of their ImageNet pre-training data, which allows both models to quickly adapt to specific classification tasks.

### Specifications:
- In the absence of abundant training data, we chose to freeze all of ResNet’s layers except for the fully connected layer, which we trained using our augmented dataset.
- Including both original and augmented data, the final model was trained on 4,000 COVID-19-negative samples and an equal amount of positive samples to ensure that accuracy was not biased towards either class.
- Our final model was trained for 18 epochs, at which point we observed a plateau of both train and validation loss. A batch size of 32 was used to pass data into the model.
- Both Adam and stochastic gradient descent (SGD) were utilized as the training optimizer, and overall we found Adam gave the most consistent results, outperforming SGD by a validation accuracy of between 5-10% past 15 epochs.
- To prevent overfitting and improve our model’s ability to generalize, we also utilized dropout, learning rate scheduling, and early stopping.

## Usage
Run the frontend using:
```sh
cd client
npm install
npm run dev

pip install -r requirements.txt

./venv/bin/uvicorn server.main:app --reload
```
Then run the backend using:

```sh
pip install -r requirements.txt

./venv/bin/uvicorn server.main:app --reload
```

## Results
Our best model accuracy thus far is: **65%**

## References
1. Ioannidis JPA. Global perspective of COVID-19 epidemiology for a full-cycle pandemic. Eur J Clin Invest. 2020 Dec. doi: [10.1111/eci.13423](https://doi.org/10.1111/eci.13423).
2. Public Health Agency of Canada COVID-19; Descriptive epidemiology of deceased cases of COVID-19 reported during the initial wave of the epidemic in Canada, January 15 to July 9, 2020. doi: [10.14745/ccdr.v46i10a06](https://doi.org/10.14745/ccdr.v46i10a06).
3. Talic S, et al. Effectiveness of public health measures in reducing the incidence of covid-19, SARS-CoV-2 transmission, and covid-19 mortality: systematic review and meta-analysis. 2021 Nov 17. doi: [10.1136/bmj-2021-068302](https://doi.org/10.1136/bmj-2021-068302).
4. Halliday T, et al. Financial Implications of COVID-19 Polymerase Chain Reaction Tests on Independent Laboratories. 2022 Aug. doi: [10.1007/s11606-022-07676-1](https://doi.org/10.1007/s11606-022-07676-1).
5. Gheisari M, et al. Mobile Apps for COVID-19 Detection and Diagnosis for Future Pandemic Control: Multidimensional Systematic Review. 2024 Feb 22. doi: [10.2196/58810](https://doi.org/10.2196/58810).
6. Madhurananda Pahar, et al. COVID-19 detection in cough, breath and speech using deep transfer learning and bottleneck features, Computers in Biology and Medicine, Volume 141, 2022, 105153, ISSN 0010-4825, doi: [10.1016/j.compbiomed.2021.105153](https://doi.org/10.1016/j.compbiomed.2021.105153).
7. Loey M, Mirjalili S. COVID-19 cough sound symptoms classification from scalogram image representation using deep learning models. 2021 Dec. doi: [10.1016/j.compbiomed.2021.105020](https://doi.org/10.1016/j.compbiomed.2021.105020).
8. Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019. doi: [10.48550/arXiv.1905.11946](https://doi.org/10.48550/arXiv.1905.11946).
