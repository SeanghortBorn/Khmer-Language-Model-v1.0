## **Khmer Language Model for Handwritten Text Recognition on Historical Documents**

Preservation of historical documents is a critical responsibility that we cannot disregard or they may vanish in a matter of days. A Study of **Khmer Palm Leaf Manuscripts Digitization** was adopted to provide public access to the Khmer Palm Leaf Manuscripts, or Khmer Sastra Sluek Rith, on the internet in order to contribute to the preservation of these priceless records that are vital to Cambodians and researchers. 
**Khmer Handwritten Text Recognition on Historical Documents** is a part of the above research which focus on creating a model that has ability to correct Khmer misspelling words that are extracted from the Sluek Rith set.

In this project you will notice that there are different models that are used with the different purposes. Each model has their description as below:
- `model_en_de.py` is implemented with Encoder-Decoder architecture.
- `model_bert.py` is implemented with BERT model.
- `test.py` is a file for testing function or data.

### Environment Setup
- Python version = 3.7.10
- PyTroch version = 1.8.1
- Conda version = 4.11.0
- Dataset sources:
  - `SBBICkm_KH.txt` : https://sbbic.org/2010/07/29/sbbic-khmer-word-list/
  - `SluekRith-Set.txt` : https://github.com/donavaly/SleukRith-Set

### Experimental Results

Below is a table of our experimental results with the different variables.

|       Models       | Original Dataset |  Dataset Size (Words)  |  Hidden Size (Layers)  |  Learning Rates  |  Epoch Size  |  Accuracy (%)  |
|:------------------:|:----------------:|:----------------------:|:----------------------:|:----------------:|:------------:|:--------------:|
|  `model_en_de.py`  | `SBBICkm_KH.txt` |          1000          |          128           |      0.001       |     1000     |     82.70      |
|  `model_en_de.py`  | `SBBICkm_KH.txt` |          1000          |          512           |      0.001       |     1000     |     94.60      |
|  `model_en_de.py`  | `SBBICkm_KH.txt` |          1000          |          512           |      0.001       |     5000     |     98.10      |
