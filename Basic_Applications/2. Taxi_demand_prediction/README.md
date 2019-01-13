Our objective is to predict the number of pickups as accurately as possible for each region in a 10min interval. We will break up the whole New York City into regions, that we will discuss later in the blog. Now, the 10min interval is chosen because in NYC one can commute 1 mile in approximately 10 minutes given the traffic is normal at that particular time.

**Datasets**

There are total 19 features in our dataset.


**Libraries**
1. DASK: Our data is very large in size. A single “csv” file is of more than 1GB in size, therefore loading all of the data at once using “Pandas” library will take up all of the RAM, and subsequently it will throw memory error. In order to overcome this problem, we are using dask library.

Instead of loading all of the data at once in a RAM, dask load blocks of a file into RAM. It loads only those blocks of file that are required right now. As soon as the processing on the currently loaded block is done, it empties the RAM and load another block of file.