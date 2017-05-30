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

### Example of generating instances with 1-second threshold

```bash
cd data/code/
./gen_instances.py Watch 1
```

Then the file **data\_Watch\_1.json** will be created in __data/__
The first arument could be *Watch* or *Phones*

### Format of Processed Files

```
{
  "user": ["a","gear","gear_1"],
  "label": "stand",
  "begin_time": 27920678471000,
  "end_time": 27921678471000
  "gdata": [
    {
      "data":["-0.16218652","-0.022104237","0.05965481"],
      "time":27920678496000
    },{
      "data":["-0.18322548","-0.06178534","0.012516857"],
      "time":27920681926000
    },
    ...
  ],
  "adata": [
    {
      "data":["-0.5650316","-9.572019","-0.61411273"],
      "time":27920678471000
    },{
      "data":["-0.83258367","-9.713276","-0.60693014"],
      "time":27920681910000
    },
    ...
  ]
}
```
