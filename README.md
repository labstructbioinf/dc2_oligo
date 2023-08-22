# DC2_oligo
## **Fast and accurate prediction of oligomerisation state of coiled-coil domains in protein structures generated using colabfold**


### **Requirements and installation** ###

1. Clone repository
```
git clone https://github.com/labstructbioinf/dc2_oligo
```

2. Install all dependencies

```
pip install scikit-learn pandas joblib ...
```

3. Check if everything works using pytest

```bash
cd dc2_oligo
pytest
```

### **Usage** ###

```bash
python src/predict.py --colabfold_output_dir DIR --use_pairwise BOOL

 ```
 | Argument        | Description |
|:-------------:|-------------|
| **`--colabfold_output_dir`** | Colabfold output directory with saved embeddings via --save-representations option |
| **`--use_pairwise`** | Flag for using sorting pair representations

```bash
python src/predict.py --colabfold_output_dir tests/data/7 --use_pairwise True
```
