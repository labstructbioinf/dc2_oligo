# DC2_oligo
## **Fast and accurate prediction of oligomerisation state of coiled-coil domains in protein structures generated using colabfold**


### **Requirements and installation** ###

1. Clone repository
```
git clone https://github.com/labstructbioinf/dc2_oligo
```

2. Create virtual environment (using e.g. conda) and install requirements

```
conda create -n dc2_oligo
cd dc2_oligo
pip install .
```

3. Check if everything works using pytest

```bash
cd dc2_oligo
python -m pytest
```

### **Usage** ###

```bash
python predict.py --cf_results DIR --save_csv STR

 ```
 | Argument        | Description |
|:-------------:|-------------|
| **`--cf_results`** | Colabfold output directory with saved embeddings via --save-representations option |
| **`--save_csv`** | Save csv by input filename (optional)|

```bash
python predict.py --cf_results tests/data/0 --save_csv testoutput.csv
```

### **Additional information** ##

For optimal usage, enter sequences that contain __coiled coil domain only__. You can easily detect such domain with our main  [__deepcoil__](https://github.com/labstructbioinf/DeepCoil)  predictor.
Please use  AlphaFold2 multimer embeddings (**alphafold2_multimer_v3**)




