# DV-antivirus
Lung segmentation and statistical analisys

## Pre-reqs
 - lungmask (https://github.com/JoHof/lungmask). Note: it does not work with pipenv. 
 - numpy
 - SimpleITK

## Usage
Clone repository.
Get elastix_linux64_v4.8.zip from NAS.
Extract in root folder.
Create params folder inside elastix folder and place .txt files there.

Run `python run.py` 

## Extras
The `organize_series` function organize a study folder in series subfolders.

## Workflow HOWTO
prima lanci python utils.py --organize PATH_STUDY_FOLDER
dove PATH_STUDY_FOLDER è il path alla cartella che ha il nome dello studio (tipo /home/mattia/covid/S1 N888999/ )
che te la organizza in serie

poi lanci python run.py --dicomdir PATH_DICOM_SERIE --outfolder PATH_OUTPUT
dove PATH_DICOM_SERIE è il path al folder della serie con mezzo di contrasto
e PATH_OUTPUT è il path alla cartella dove vuoi l'output (il csv con i valori)

puoi usare --force_cpu come flag per usare la cpu se la gpu ha problemi di memoria

infine lanci python utils.py --examine PATH_CSV per generare l'istogramma (te lo visualizza e lo salva in plots.png nella root da cui lanci)
