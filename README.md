# DC2_oligo
## **Fast and accurate prediction of oligomerisation state of coiled-coil domains in protein structures generated using colabfold**


### **Requirements and installation** ###

1. Clone repository
```
git clone https://github.com/labstructbioinf/dc2_oligo
```

2. Install all dependencies

```
conda env create -f  environment.yaml
```

3. Check if everything works using pytest

```bash
cd dc2_oligo
pytest
```

### **Usage** ###

```bash
python predict.py --cf_results DIR  --save_csv STR --predict_topology BOOL

 ```
 | Argument        | Description |
|:-------------:|-------------|
| **`--cf_results`** | Colabfold output directory with saved embeddings via --save-representations option |
| **`--save_csv`** | Save csv by input filename (optional)|
|  **`--predict_topology`** | (Experimental) Predict topology of oligomer (optional)|

```bash
python predict.py --cf_results tests/data/0 --save_csv testoutput.csv --predict_topology
```


