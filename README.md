# Update
I've made changes to the UseDataSynthesis notebook to convert a whole batch of files, and to the datasynthesis.py file to solve some issues generating data
One of the checks which 'hung' things was how much percentage of data each subject has. If a subject has NO data whatsoever, it'll not be considered for data modeling. BUT - there is numerical and there is categorical data. Categorical data is never used, and once percentage categorical > numerical, then subject might be thrown out which should not have been. This is now fixed.

Another issue is regardign debugging. I use pudb, and use remote terminal to look at code. BUT - python 3.12 and higher don't work very well. 
Therefore, conda environments (or at least, those that need debugging) are ideally python=3.11. This is not very future-proof.



# Conda
Here some installation instructions to set up your conda environment.
Better to use miniconda - install what you need, and not the bloaty anaconda installer

```python
conda config --add channels conda-forge
conda create -n shield python=3.11
conda activate shield
   
conda install scikit-learn matplotlib jupyterlab
conda install numpy pandas
conda install python-kaleido # not needed for synthesis, but for our other module - data exploration and model creation
conda install anywidget    # same

pip install pudb  # the debugger
pip install tqdm  # the progress bar
```



# Data Synthesis
This is small python module to create a synthesized version of existing participant data for upload with collaborators. This will not share any of the existing data itself. The synthesized data will ideally allow the development of models and data analysis methods that would in the end also be suited to use on the actual existing data.

In addition, the module already lays a small foundation to come to a consensus about how data should ideally look like, across different instututions. 

# Synthesis procedure
This module can load in a .csv data file, in long format, and it will read and create an identical copy of the input data sctructure. Ever cell of the output structure is filled with pandas NA. Then, the actual data file is read in, particiant ID's are randomized, and new data will be generated based on some averages of the existing data. Within the synthesized data, Within-subject variance and between-subject variance are maintained.

With time-series data, we rely on a model that uses phase randomization of the original timeseries to synthesize new timeseries with similar attributes. With Single-entry data (possibly combined with categorical data), we keep the categorical data constant (as this usually has no information in it anyway), and the entries are analyzed. A PCA model is fit and we will impute/synthesize data that maintains the origoinal's session/measure covariance. - basically by putting that all into a single vector for each participant before PCA analysis.


# Bookkeeping
- This module works on long-format data; we also do have matrix-format data. In order to handle that, I'll write a matrix-to-long format (and back again) converter. We can then extend this functions to handle data from other sites.
- A useful feature (see below screenshots) - is that it can output what is IN the data to begin with.

# Python Requirements
- numpy
- scipy
- pudb - pip install pudb - for debugging
- tqdm - pip install tqdm - the progress bar


# how to use
This works on long-format data; i.e. each value of each measurement is 1 complete row. There is also 'normal', or matrix format, but I will write something to convert matrix to long format and back again to help out.

You need to know:
- the full path to the file (or relative to where you use python)
- the column name where participant id's are stored
- the column where the data values are
- the column where the variable TYPE is 
- the column where the variable UNIT is 
- cols_to_keep - can be empty, but allow you to specify which other columns should be copied into the synthesized output; best not to use identifiable columns here.
- the column where the session is (i.e. time point 1, 2, pre, post, etc etc). Can also be NA
- the column where time values are
- the column where time units are

The python module needs all of this to know where the participant numbers, sessions, and data types are; and what to look for. If there is one spelling error, it'll hang.



```python
import datasynthesis

ds = datasynthesis.DataSynthesis(seed=1)

ds.load('6_Physiological/2_V02/3_heat_physiology_thermoregulation_hr.csv',
        participantid='participant_id',
        data='value',
        type='variable',
        unit='unit',
        cols_to_keep=['trial', 'sex'],
        session='period',
        time_value='timepoint_value',
        time_unit='timepoint_unit')
        
ds.checkit()
        
ds.synthesize()

ds.export()
```


![Here is a screenshot of how it is used](figs/example_screenshot.png)
