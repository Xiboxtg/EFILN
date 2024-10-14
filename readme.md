# EFILN

This is the code of the EFILN, used to invert electric field information to obtain the position coordinates.

## Version Information

The versions of the code usage library are listed below:

|  Library   | Version |
| :--------: | :-----: |
|   numpy    | 1.24.3  |
|    gym     | 0.26.2  |
| matplotlib |  3.7.5  |

Torch version for the code is 2.2.2 with Python 3.8.19.

## Code Function Description

The code function descriptions are listed as follows:

*****

```python
dataset_generation.py # file for EFILN train
```

```python
network.py # network definition and structure setup of EFILN.
```

### Running Experiments

*****

To train the model: 

~~~python
python .\dataset_generation.py
~~~

