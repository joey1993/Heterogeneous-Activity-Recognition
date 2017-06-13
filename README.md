# Heterogeneous-Activity-Recognition
UCLA CS 239 Class Proj

## Presentation Slides

The [slides](https://docs.google.com/a/g.ucla.edu/presentation/d/1HavhKL3Ix8TY1fMxwHfMI023mfTtTPyNsAoevFGJkO0/edit?usp=sharing) is located in Google Slides.

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

### Example of generating features with data_Watch_1.json file

```bash
cd data/code/
python ./gen_feature.py data_Watch_1

```
Then the file **data\_Watch\_1\_feature.json** is located in __data/features/__

Note that some of the features are empty or contain NaN values. The order corresponds to the orignal data file. 
### Format of feature Files
```
{
  "features_a": [-6.5836123204347823, 3.7443888264980578, 14.020447684003502, -9.1512375000000006, -0.56503159999999997, -12.600683, 0.46803190885932655, -1.3721246539033078, 6.653786049999999, -6.5949587478260865, 3.1915007938641113, 10.185677317235253, -4.5567646000000002, -3.6056678, -11.08276, -0.351418609378596, -1.7602101423374787, 6.3060282499999998, -0.35049031826086957, 0.55090349273103645, 0.30349465830325512, -0.60693014000000001, 1.0157400000000001, -1.1791444, 0.8800286945230752, -0.2142537480953135, 0.94600892500000011, -0.85198310073033634, 0.22425757757377504, -0.36379276342804123], 
  "features_g": [-0.012214225918181816, 0.5255247959068664, 0.27617631111295354, -0.017310547499999999, 0.81039994999999998, -2.0383835000000001, -2.188110427784416, 7.6546374560268955, 0.18861838624999999, -0.17363810760909093, 0.19743298014370372, 0.038979781648424111, -0.085221155000000007, 0.027696875999999999, -0.55287224000000001, -0.7990249037975988, -0.8345142828829073, 0.29687587679999999, -0.20029392159090909, 0.48855979813970313, 0.23869067635830746, -0.090813790000000005, 1.1475561999999999, -1.2319783, -0.2743844132046165, 2.2487279813367627, 0.12536830874999999, -0.24262115779401275, -0.92592795888927704, 0.52983472334504977], 
  "label": "stand", 
  "user": ["a", "gear", "gear_1"], 
  "begin_time": 27920678471000, 
  "end_time": 27921678471000
}
```

### Example of genenrating a smaller dataset with only slots containing 200 points

```bash
cd data/code/
python gen_instances_small.py data_Watch_1 200
```

Since time point distribution is diverse, we only keep slots with more than 200 points.
200, the default number, can be changed according to demands.

### Example of splitting dataset into Training and Testing parts

```bash
cd data/code/
python gen_feature_train_test.py data_Watch_1_small 0.2
python gen_feature_train_test.py ./feature/data_Watch_1_feature 0.2
```
Splitting ratio was set to be 0.2



