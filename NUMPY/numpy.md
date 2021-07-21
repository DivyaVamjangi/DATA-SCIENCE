
You must be wondering why one should read this article? The answer is straightforward. There are different data preprocessing libraries for which numpy is the basis, so we try to discuss the complete data pre-processing just using numpy

  

We will also walk you through plenty of numpy universal functions and their use in real-world applications.

  

You can download the notebook from this [link](https://drive.google.com/file/d/1knzRc5XdWyAbOxdY0CmUAJiFz0_Yoc9Y/view?usp=sharing).


  

Let’s cut to the chase,

  

We’ll be performing all of the below listed steps.

  

1. Importing & Loading the data

2. Exploring 

3. Cleaning 

4. Stiching(getting things together) and saving the data

5. Loading 

  

# Importing & Loading

  

1.1 IMPORTING

  

We Import NumPy library as np where np is just an alias, anything can be used in place of np. 

  

![](https://lh5.googleusercontent.com/qR1AtKWT-TZjS-Gtw4n4bGSzz7a5IgfzyvmfYH6B2phFqSq8ZDMHtSbvp1tagPZIpt8S0APNKA_FbwWQjpp4Kl8Ca2Jo56-sMFxrRVtRo5R2OOjeZuY9T3UvJ5oQsKXAJ7wzq4yl)

  

We can also set print options to determine the way floating-point numbers, arrays, and other

  

NumPy objects are to be displayed. This is just optional for a better view of your data.

  

1.2 LOADING

  

Numpy provides us two different functions to load the data, [$np.loadtxt()$](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html) and [$np.genfromtxt()$](https://numpy.org/doc/1.20/reference/generated/numpy.genfromtxt.html).

  

Here we will be using $np.genfromtxt()$ as it can handle the missing values in data as specified.

  

![](https://lh6.googleusercontent.com/koNShW7JbDSigHbHXj-lgfn7pT8r8eoXEEc4T7FiUFUtGFQcWBvcuhszd1cmQ0t2QN-ZzO4NRfuYLT5wLQcNd4kYWEvP1Kd5TAWbQ_4RV3JHT0cpCKwCzwn4xLRUvKmleeDOgPT9)

  ---ADD PARA--

  

# Exploring

  

In this section, we shall have a bird view of our data and see what needs to be done. This is kind of EDA but we do not visualize the output with graphs etc. As the concept of the article is to just use Numpy for data preprocessing.

  

--Write some more description of how the data looks--

  

2.1 Splitting

  
As we can see, only floating point variables are displayed and rest are nan. This is because the $np.genfromtxt()$ function uses dtype=float as default which is why string columns are converted to nan’s. After all, they're not a number.

So, splitting is required to access them separately. We perform that by taking the mean of each column and if the mean is nan, it refers to the string columns as they were filled with nan values while loading the data, and the remaining are the numeric columns.

 

[$np.nanmean()$](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html) to compute the arithmetic mean along the specified axis, ignoring nan’s.

  

![](https://lh4.googleusercontent.com/GsnT5w0ToDWQo3f2O9-2AdpJNXQMIBIkBlsjxgUZEnBWKwyTYyLFS0QBqZh5pbt-YPkbmw7WZUUQrhkx5I38Qbb4E0PefNlJs6H2gPeuugm6SPcdu1l3USH2HWecRxAPb0utHppo)

  

[$np.argwhere()$](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html) is used to find the indices of null(for string columns) and non-null(for numeric columns) values in columnwise_mean.




![](https://lh5.googleusercontent.com/L_aqZsy4NzCKfa1zmSlHBOZMUVGxned3L150mN7rReb1NGlW63-2JTBzpbm1Srvy18GGd7sbaBxrSv7R_TvhanptjaLazIFAV7spgOa8lxyLrnszP_0lAKkg_l_KFH45E8C03spu)

![image](https://user-images.githubusercontent.com/53438169/126495174-05633d78-6bd3-4547-980c-9c72ea6b0721.png)

![image](https://user-images.githubusercontent.com/53438169/126494999-61e8596b-934f-47c2-91d9-7d832648bf2c.png)  

On top of it apply [$np.squeeze()$](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html) to remove single-dimensional entries from the shape of numeric and string columns.

  

![image](https://user-images.githubusercontent.com/53438169/126495281-9b810e8b-c014-4926-8066-c4f38878a5db.png)


  

2.2 Accessing

  

Re-Import data by specifying data type and indices of columns to consider using dtype and usecols parameters respectively in genfromtxt().

  

Remember to skip the header for numeric columns as they are string values and including them will result in the occurrence of nan values.

  

![](https://lh3.googleusercontent.com/d11zJsmb5WCoXJLhp4l0qn37YqNdmMWKt-UjHh7TP-4KAoq-JBMWzTL-ka5blOjKd7NPKjavUXXPfUf1prLPT_SnRpB7LfJ4uMSI-6KHLAe3lND88-tOUlXI8H3nktpgm_9yQknX)

  
----ADD PARA----
  

3. Pre-processing The Data

  

Now, we know how the raw data looks, so we shall perform some preprocessing/data engineering to convert the raw data into the format we want.

  

3.1 For String Columns-

  

3.1.1 Handling the Missing Values-

  

As it can be seen our dataset is having lots of missing values that need to be handled. to do this,there are several ways –

  

![](https://lh3.googleusercontent.com/Ry7GLyEabcQvrkwAIdJQF91VZU7iPJlDCteu85sWgcdn74wtpQrFwgkkH1m2NDY7avXGiKJzgK9Xvj-8LInObkN_jl0dibdXsFXdJsa2_NrkETW_AypIyiiMG7w8N93ryRX0keWM)

  

· We can ignore them and delete the entire column

  

· We can set a threshold value above which we delete the entire column

  

· We can replace the missing values with mean, median or mode.

  

In our case, we cannot ignore the missing values(because???). So, we set the threshold above which the entire column will be deleted and if missing values are less than the threshold value, we replace it with the mode(why? Just for the sake of article) of the column.

  

Firstly, we set our threshold value (in our case 50%) and count the number of missing values in each column and check if the count exceeds that threshold, we store the indices of those columns in a list.

  

![](https://lh6.googleusercontent.com/DMcNVNMP5L_nvlbeyInadWrImt5rndBFJa1NsJIVM4ZeqnSIq7XoJ0qPG2QafVCuWYmwgAOUDuhvhY_se8xjuYtuVib65PpdF1b4TdBKzU5ar1KOp7sVxSnYHTEW-EB-B9RnyjFW)

  

And then we delete those columns from the dataset using [$np.delete()$](https://numpy.org/doc/stable/reference/generated/numpy.delete.html) but before deleting those columns we reverse the list because if we delete from start then other column indices are changed and cases might occur where wrong is deleted.

  

![](https://lh6.googleusercontent.com/l6DtGlWm_pb7Yz8yPul8_fsZkQrTtYeRciExdvo2mldWx265qrKxdFR_3rNPQVahZtAiYAYl-DpOuQvtHJCz_lmopA6qyLcPKNtElYi-kq5ae3TK552cKmNvNBflKNmUHnsDp2-Q)

  

Secondly, for replacing the missing values with the mode of the columns we first find the unique element and its count using [$np.unique()$](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) and we search for the element with maximum count and replace all the missing values with that element.

  

![](https://lh3.googleusercontent.com/64q3WPUox-dEcWXohsgeMFDGkT9wE3YjPO5jL7HzIwZ2P23Waujuupymhenc4m4Reg2prF9JGc2TYT0O87O4Ke17ZBdHF4Ty_fKYRirTWBoDDbpRpJV8bzbZYPIxtCkVJ0PgwtoR)

  

  

3.1.2. Encoding Categorical Data

  

Now, before moving on, let’s separate the headers from the data and store it in header_string as we don’t want to encode the headers of the columns.

  

![](https://lh3.googleusercontent.com/Wp7weWJRZpkn-3QbUH6nr03jowwORYfXjyCQ5WNs8UhDvplR4RlqCmFikdoJoykkTalUB3ShO4OZH2BnGl3-fpnDwhfpRWRNfpoYnGPxPAzW14oWQKBZJjB4gGJ-0tmtEt6s6wkd)

  

  

For each column we will find out the unique values in the column and we make them as keys and give numeric values to those keys starting from 1 and then zip the keys and values into the dictionary.

  

After that we traverse each column and replace the elements in the column with dictionary values.

  

![](https://lh5.googleusercontent.com/rsppnSb9qnLAqqHtVWG_a_J5gwhj5IGbgPVdJWraCtBq4Kwp-NtGYelUebDRIsToVfnuEg085r8GSrYrklExiCe1uX2mTcmcPTMI3Fh4G_EKBANnh7vOOKjDZJYZLgPw175sLliy)

  

![](https://lh5.googleusercontent.com/zwNg0IS75yS1CoAvxYfecGni9FKyZ_CvlF5dI6egLoJ3fc4aZu2oKs0F2V_7L-4qKHtehoFwBBojTu_cVuNsFFTP52uFfq9OADlvzraiKZgFU9PEQpNJXxJjf0deOsA-CdtIdjiA)

  

  

3.1.3 Type Casting

  

Although we have encoded categorical values to numeric values it can be seen the values of our data are still string type. Therefore, we need to typecast the string values to float using [$np.astype()$](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) function.

  

![](https://lh6.googleusercontent.com/GNowCVaPx9J4R74wELhSIOdgwDpRGz_FQZl3xIBPCzvEKbpRcaioc-5DsIdCy9_kLsTY-X91eq9yK6OqoY2W6P9cvEqCrLOgeyCBQ1P8hpNaa02WpDB6xWyjmyTAUilYxAuOGW9w)

  

3.2 For Numeric Columns

  

3.2.1 Handling Missing Values

  

For counting the missing values in the numeric column we can use [$np.isnan()$](https://numpy.org/doc/stable/reference/generated/numpy.isnan.html).[$sum(axis=0)$](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) which gives us the count of missing values in each column.

  

![](https://lh4.googleusercontent.com/j_7BWWH_XiX0LW57Dq9t5qsp_k_RhvNVaJfVynuGuTg26hS62Suxnm8Jtr5kvsrTN-Ajog6mPV86YhzG-Ro6MJ6wdntIg7strBP9JLpwx5P7bYZa4TUuLQdj8CHTHD86MDUVnOGK)

  

As we can see that there are no such columns in our dataset where missing values are greater than the threshold Therefore no deletion of columns is required in our case.

  

Although we need to handle those missing values, we take the mean (just for sake of article) of each column using np.nanmean(). It computes the arithmetic mean along the specified axis , ignoring the NaN’s.

  

We then find the indices of the column with NaN’s values using np.where() and finally we replace the NaN’s with the column mean respectively and align the array using np.take().

  

![](https://lh4.googleusercontent.com/kNccI0WTexs9nMYP5Z_alD814caj-w0qzgOnxdbbISEiE0A1bvR1ZKz9stkHPiXGIH7O-OfFtUE25nSWkjZw6vVJ24ulEQwHx7PUhRGAZt5heTGT91ivIXIrNWfHThZunb9amDNX)

  

  

4. Stitching (Getting things together) and Saving the data-

  

We now reload the numeric columns to get the headers of the numeric columns.

  

![](https://lh6.googleusercontent.com/TAXXrHt2crxwwsmZ-LYiPmUZO13lCzeKGTzM0hs8vthPdFU_lx0rmlvalSzPfeiiVdzSYjjhFAzPxO-2Wa0WiDbfmFprvre80N958AcZhmzPYTPruH8MxSrH-FfZr79lk4bDUc1Z)

  

Now that we are done with data preprocessing we have header_string(containing header of string columns), header numeric (containing header of numeric columns), data_string(containing pre-processed string columns) and data_numeric(containing pre-processed numeric columns).

  

Finally, we put all the preprocessed data together and create the checkpoints function to save all the data into a single file and for that we use [$np.savez()$](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) to avoid pre-processing the data again when we need it.

  

![](https://lh6.googleusercontent.com/OcmTgxOiZWaav_bg9tgoZe9aZVQjA6kEkhtAKHaKxXFZu1FZih8U5giC3kvgpNhlLuz4EoIi_1yk1CFuccUFW9-Whg-CTAP2zg0tM5EQEgm-JJ0Ftqzw-H89stoKE2dN3krTwhNj)

  

5. Loading the data

  

How to load the checkpoint files we just created?

  

For loading the data from the files we use [$np.load(filename)$](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

  

![](https://lh6.googleusercontent.com/T2Bb4J-91_edfhTW_OgIMe1ioy6YjjOzqXPnwiy7Gtjq45k3FKBq3YfHhsYK_GHEjadOoB1WuUfLRCBqSDoa5wayIetlhqBsA14Zir9_HvAxem3uxp9y-6iRUW2WbKNGAYmwB5SO)

  

![](https://lh5.googleusercontent.com/riovPdirHv-l6RlMt1v0znVms1lPkTU82XbVnMYBXt6WLgZHFf7Qq4mgEgDo0_rOZUfqFVZwNZVsxnyUGPsYORwhNGIETLv-ybSVU1tmk6xn4vTKoTsJfCCAUbnWcohoVDDQVNXg)

  

  

Now our data is finally ready for training!!

We hope this article was helpful in understanding data preprocessing. Be sure to drop your feedback and suggestions below.

Thank You for reading!


