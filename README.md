# Semantic Typing of Event Processes
This is the repository for the resources in CoNLL 2020 Paper "What Are You Trying Todo? Semantic Typing of Event Processes". This repository contains the source code and links to some datasets used in our paper.

Environment:

    python 3.6
    Transformers (Hugging) version 2.11.0 (Important)
    PyTorch
    nltk
    AllenNLP
  
## Dataset  
./data contains the wikiHow Event Process Typing dataset contributed in this work. The same folder also contains verb and noun glosses from WordNet, and the SemCor dataset used for WSD.  
The raw file of wikiHow Event Process Typing dataset is given as data_seq.tsv, where each row records the content and types labels of a process. Specifically, each tab separated row contains a sequence of subevent contents, and the last two cell are the action and object labels.  
The binary file is a saved instance of the data.py object in utils, which has already read the process data and label glosses, and provided necessary indexing information to split, train and test.  
./process archives several programs for dataset proprocessing.  

## Run the experiment  
The program ./run_joint/jointSSmrl_roberta_bias.py runs the experiment for training and testing with excluded 10\% test split. It should execute with the following pattern  

    python jointSSmrl_roberta_bias.py <skip_training> <alpha> <margin_1> <margin_2>  
  
For example:  

    CUDA_VISIBLE_DEVICES=4 python jointSSmrl_roberta_bias.py 0 1. 0.1 0.1
  

## Console demo application  

./run_joint/console_roberta_bias.py is a console application where the user can type in event processes and obtain the multi-axis type information on-the-fly.  Simple run this program, wait until it loads a pre-trained model, and type in an event process where subevents are separated by '@'. For example, the following input   

    read papers@attend conferences@go to seminars@write a thesis
  
would receive type information such as  

    [('get', 0.6021211743354797), ('retain', 0.6217673718929291), ('absorb', 0.6397878527641296), ('pass', 0.6577234268188477), ('submit', 0.6673179864883423), ('present', 0.6688072383403778)] 
    [('doctorate', 0.5141586363315582), ('psychology', 0.5413682460784912), ('genetic', 0.5501004457473755), ('science', 0.5507515966892242), ('determinism', 0.5621879994869232), ('grade', 0.5723227560520172)]

Link to the pre-trained full models for console demo: https://drive.google.com/drive/folders/1b8peVVRNANL7r_Wnyyt4pPsyNROIlOfT?usp=sharing  
Users can also train the model on the full wikiHow event process dataset by running ./runjoint/train_full_roberta_bias.py  

We are working on creating a Web system demonstration, and would update at here soon.

## Reference
This work is to be published at the 24th SIGNLL Conference on Computational Natural Language Learning (CoNLL), 2020.  
Bibtex:
  
    @inproceedings{chen-etal-2020-what,
      title = {``What Are You Trying To Do?'' Semantic Typing of Event Processes},
      author = "Chen, Muhao and Zhang, Hongming and Wang, Haoyu and Roth, Dan",
      booktitle = "Proceedings of the 24th Conference on Computational Natural Language Learning (CoNLL)",
      year = "2020",
      publisher = "Association for Computational Linguistics"
    }


