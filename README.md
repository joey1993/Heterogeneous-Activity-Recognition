# Heterogeneous-Activity-Recognition
UCLA CS 239 Class Proj

## Preprocessing

### Raw data placement

Unzip the [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip) and put all files in the directory __raw/__

### Example of sorting raw data

```bash
cd raw/code/
./sort_by_user_time.py ../Watch_accelerometer.csv
```

Then the file **Watch_accelerometer.csv.sort** will be created in __raw/__

### Example of generating instances with 1 second threshold

```bash
cd data/code/
./gen_instances.py Watch 1
```

Then the file **data\_Watch\_1.json** will be created in __data/__
