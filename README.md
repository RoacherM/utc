
# UTC

**UTC** stands for "Universal Time Expression Converter."

This repository provides a simple encoder-decoder architecture designed to convert time expressions into standard formats (for example, from "twenty and twenty three, may first" to "20230501").

We employ BERT/Roberta as the encoder and a single direction GRU as the decoder. The CLS token from the encoder is used as the initial input for the decoder, while the last hidden state of the encoder serves as the initial hidden state for the decoder. The decoder then continues decoding until it reaches the desired length (in the example provided, the sentence is decoded into the "YYYYMMDD" format, a total of eight characters).

Parsing time expressions can be quite intricate. To simplify the process, it may be beneficial to incorporate some preprocessing steps before parsing the data. For instance, using an NER model to extract the date-related SPAN, standardizing word forms (like changing "second" to "two"), and then parsing with the model. Unless otherwise stated, it's assumed that training is conducted on the data with standardized word forms.

PS: GPT4 is deeply involved in this repo

**Training:**

```python
python train.py
```

**Testing:**

```python
python test.py --spoken_dates "may first" "dec twelfth"

## [(array([0, 0, 0, 0, 0, 5, 0, 1]), 'may one'), (array([0, 0, 0, 0, 1, 2, 1, 2]), 'dec twelve')]
```
