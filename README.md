# Score! Accompanying code to the paper: "Predicting Sex and Stroke Success – Computer-aided Player Grunt Analysis in Tennis Matches"

*Professional athletes increasingly use automated analysis of meta- and signal data to improve their training and game performance. As in other related human-to-human research fields, signal data, in particular, contain important performance- and mood-specific indicators for automated analysis. In this paper, we introduce the novel data set SCORE! to investigate the performance of several features and machine learning paradigms in the prediction of the sex and immediate stroke success in tennis matches, based only on vocal expression through players’ grunts. The data was gathered from YouTube, labelled under the exact same definition, and the audio processed for modelling. We extract several widely used basic, expert-knowledge, and deep acoustic features of the audio samples and evaluate their effectiveness in combination with various machine learning approaches. In a binary setting, the best system, using spectrograms and a Convolutional Recurrent Neural Network, achieves an unweighted average recall (UAR) of 84.0 % for the player sex prediction task, and 60.3 % predicting stroke success, based only on acoustic cues in players’ grunts of both sexes. Further, we achieve a UAR of 58.3 %, and 61.3 %, when the models are exclusively trained on female or male grunts, respectively.*

## 1. Cut audio snippets of relevant parts of the videos and add annotations

### Usage
```
usage: cut_relevant_audio_snippets.py [-h] -a ANNOTATION_PATH [-m MP4_PATH]
                                      [-o OUTPUT_PATH] [-g GENDER]

Cut mp4 regarding the annotations

optional arguments:
  -h, --help            show this help message and exit
  -a ANNOTATION_PATH, --annotation_path ANNOTATION_PATH
                        annotation text file folder
  -m MP4_PATH, --mp4_path MP4_PATH
                        corresponding video file folder
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output path
  -g GENDER, --gender GENDER
                        "f" for female, "m" for male
```
### Requirements
- non-off-the-shelf python packages: moviepy.editor, pydub
- Unfortunately, our university does not provide a permanent storage for large data, hence, annotations, video_ids and trained models upon request.
- Make sure that it is legal to crawl videos in your country according to the fair use principle for research.

## 2. Extract multiple features from the audio snippets
Representaitons are extracted and stored as .csv in ./features/$feature_type$/. 
### Usage
```
usage: extract_features.py [-h] -f FEATURE_TYPE [-l LABEL_TYPE] [-t SPLITS]
                           [--holdout]

Prepare feature .pkl

optional arguments:
  -h, --help            show this help message and exit
  -f FEATURE_TYPE, --feature_type FEATURE_TYPE
                        specify the type of features you want to generate
  -l LABEL_TYPE, --label_type LABEL_TYPE
                        specify the type of label you want to generate
  -t SPLITS, --splits SPLITS
                        specify no of data splits
  --holdout             hold one partition back for test only
```
### Requirements
- non-off-the-shelf python packages: mmatplotlib, librosa, imageio, PIL
- Store the data in data/, using a separate directory for 'women' and 'men' tennis players.
- Create a directory 'tools' in the main directory and add openXBOW.jar (https://github.com/openXBOW/openXBOW).
- For the extraction of Deep Spectrum features install Deep Spectrum according to https://github.com/DeepSpectrum/DeepSpectrum. After installation activate the virtual environment before extracting features with the 'extract_features.py' and the parameter '-f ds'  

## 3. Run experiments
The .csv features are serialised and stored as data objects (./features/pkl/$feature_type$/\*.pkl) for every feature/label combination. All experiment parameters (including network hyperparameters) can be adjusted in config.py.
### Usage
```
usage: main.py [-h] -f FEATURE_TYPE [-l LABEL_TYPE] [-m MODEL_TYPE]
               [-n EXPERIMENT_NAME] [-g GENDER [GENDER ...]] [--verbose]

Prepare feature .pkl and run experiments

optional arguments:
  -h, --help            show this help message and exit
  -f FEATURE_TYPE, --feature_type FEATURE_TYPE
                        specify the type of features you want to use
  -l LABEL_TYPE, --label_type LABEL_TYPE
                        specify the type of label you want to use
  -m MODEL_TYPE, --model_type MODEL_TYPE
                        name of model type
  -n EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        name of experiment
  -g GENDER [GENDER ...], --gender GENDER [GENDER ...]
                        gender of data: all m w
  --verbose             prints more output information if true
```
### Usage Deep Spectrum
```
usage: deep_spectrum_model_training.py [-h] [-l LABEL_TYPE] 
               [-t SPLITS] [-g GENDER ] [--verbose]

Prepare feature .pkl and run experiments

optional arguments:
  -h, --help            show this help message and exit
  -l LABEL_TYPE, --label_type LABEL_TYPE
                        specify the type of label you want to use
  -t SPLITS, --splits SPLITS
                        number of splits for cross validation
  -g GENDER, --gender GENDER
                        gender of data: all m w
  --verbose             prints more output information if true
```
### corresponding files
- config.py: Configuration file including operation paths and paths to OpenSmile
- network_utils.py: Neural network achitecture
- model_training.py: Training
- measures.py: Measure calculations
- utils.py: Universal helper functions

### Requirements
- non-off-the-shelf python packages: matplotlib, librosa, imageio, sklearn, keras
- make sure a suitable backend is installed and the directory structure is as described below

## Directory structure
- ./data
  - e.g. /men/1_TTlUlYoyma8_2095/ > includes all .wav of this video
  - e.g. /women/1_VumK4lSfr_w_1030/ > includes all .wav of this video
- ./features: All extracted features
- ./experiments (created, experiment output)

## Experiments
FEATURE_TYPE, LABEL_TYPE, MODEL_TYPE work completely independent of each other. Everything is executable with any combination. Features are extracted automatically if not available.

### Examples
#### MFCC (SVM) on gender prediction.
We extracted 40 MFCCs based on the audio snippets using the Python package, Librosa. 
 ```
 python extract_features.py --feature_type mfcc -l gender
 python main.py -f mfcc -m svm -n mfcc_SVM_gender -l gender 
 ```
#### COMPaRE (SVM) on score prediction.
Extracting COMPaRE requires OpenSmile 2.3. Also Low-level descriptors are extracted (see ./features/llds/) but only used for BoAW.
```
python extract_features.py --feature_type compare
python main.py -f compare -m svm -n COMPARE_SVM_all -l point
```
#### SVM: middle, mean of features
This can be done with every feature fed into SVM. The corresponding parameters are config.py, get_svm_config(), config['svm_seq_agg'] = '....' mean or middle
```
python main.py -f lld -m svm -n COMPARE_SVM_mean -l gender 
```
