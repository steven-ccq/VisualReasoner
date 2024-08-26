|Dataset|Images|Questions|
|-|-|-|
|TextVQA|https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip|https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json|
|ST-VQA|https://rrc.cvc.uab.es/?com=downloads&action=download&ch=11&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy90ZXN0X3Rhc2szX2ltZ3MudGFyLmd6|https://rrc.cvc.uab.es/?com=downloads&action=download&ch=11&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy90ZXN0X3Rhc2tfMy5qc29u|
|TallyQA|http://images.cocodataset.org/zips/train2014.zip http://images.cocodataset.org/zips/val2014.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip|https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true|
|GQA|https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip|https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip|

## TextVQA
```bash
mkdir data/TextVQA
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip -d data/TextVQA
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
mv TextVQA_0.5.1_val.json data/TextVQA
```

## ST-VQA
```bash
mkdir data/ST-VQA
# login and download the images through website
# https://rrc.cvc.uab.es/?com=downloads&action=download&ch=11&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy90ZXN0X3Rhc2szX2ltZ3MudGFyLmd6
tar -zxvf test_task3_imgs.tar.gz -C data/ST-VQA
# login and download the questions through website
# https://rrc.cvc.uab.es/?com=downloads&action=download&ch=11&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy90ZXN0X3Rhc2tfMy5qc29u
mv test_task_3.json data/ST-VQA
```

## TallyQA
```bash
mkdir data/TallyQA
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip train2014.zip -d data/TallyQA
unzip val2014.zip -d data/TallyQA
unzip images.zip -d data/TallyQA
unzip images2.zip -d data/TallyQA
wget https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true
unzip tallyqa.zip -d data/TallyQA
```

## GQA
```bash
mkdir data/GQA
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip -d data/GQA
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip questions1.2.zip -d data/GQA
python data/process_gqa.py
```