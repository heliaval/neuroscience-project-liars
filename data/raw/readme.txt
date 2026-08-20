This dataset comprises four main directories, below is a detailed description of each directory and its contents:

1. [OneDCNN]: Contains preprocessed EEG epochs that are artifact-free, aligned either with the decision-making onset or the feedback onset. The EEG epochs within each '.mat' file are concatenated by participant, specifically prepared for OneDCNN classification tasks in Python.

2. [Preprocessed]: This directory holds preprocessed, artifact-free EEG epochs, similarly aligned with the decision-making or feedback onset. It features individual sessions, with filenames indicating the player and observer participants, organized within separate folders for each stimulus type.

3. [Raw]: Direct outputs from the BrainVision recorder, these raw EEG files are categorized by each participant's role (player or observer). Filenames are structured to reflect this, for example, Player_sub01.eeg or Observer_sub01.eeg.

4. [Behavioral log and trigger timestamp]: Contains behavioral logs and trigger timestamp information for each experimental session. The naming convention mirrors the paired participant structure, highlighting their roles and participant numbers, e.g., Player_sub01_Observer_sub02_Behavioral.txt or Player_sub01_Observer_sub02_Timestamp.txt.


Folder Structure Overview:

┌─ OneDCNN/
│   ├── DecisionMaking.mat
│   └── Feedback.mat
│
├── Preprocessed/
│   ├── DecisionMaking
│   │   ├── Player_sub01_Observer_sub02.mat
│   │   ├── Player_sub03_Observer_sub06.mat
│   │   ├── ...
│   │
│   └── Feedback/
│       ├── Player_sub01_Observer_sub02.mat
│       ├── Player_sub03_Observer_sub06.mat
│       ├── ...
│
├── Raw/
│   ├── Player_sub01.eeg
│   ├── Player_sub01.vhdr
│   ├── Player_sub01.vmrk
│   ├── Observer_sub02.eeg
│   ├── Observer_sub02.vhdr
│   ├── Observer_sub02.vmrk
│   ├── ...
│
└── behavioral log and trigger timestamp/
    ├── Player_sub01_Observer_sub02_Behavioral.txt
    ├── Player_sub01_Observer_sub02_Timestamp.txt
    ├── ...


