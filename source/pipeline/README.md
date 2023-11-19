# UFC Betting Pipeline Full Technical Documentation

## Download the data
We use the [scrape_ufc_stats tool](https://github.com/Greco1899/scrape_ufc_stats) to get six csv files containing all the ufc data on ufcstats.com. We use these to initialize pandas dataframes.

## Clean the data
This is the longest and most convoluted step. For our purposes, this cleaning section comprises every step between downloading the data and having a complete training/test set. 

### Cleaning the fight stats df

