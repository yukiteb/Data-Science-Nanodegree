# DSND-AirBnB-Data-Analysis
This project is done as a part of Udacity Data Science Nanodegree Program. In this project, we deep dive into the Airbnb data for Washington D.C. to see if we can gain any insight. As the capital of the United States, D.C. has been attracting a lot of re-development. In fact, Amazon announced recently that they will have the second head-quarter in Northen Virginia, which is just a several miles away into DC. Also, personally this is where I live and it makes the project more interesting for me.

In doing this, we will follow CRISP-DM (Cross Industry Standard Process for Data Mining), which specifies the following steps.

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Data Modeling
5. Evaluate the results
6. Delopyment

## File structure
The files are structured as follows:
```
- data
| - DC
| |- review.csv  # data containing reviews on each property (not used)
| |- listings.csv  # data containing detailed information on each property
| |- calendar.csv  # data containing availability/price for the next 1 year

- AirBnB Data Analysis.ipynb # Jupyter notebook for analyzing the data

- README.md # Read me
```


## Key Business Questions
Following the CRISP-DM, we first ask the business relevant questions. In particular, the key questions we would like answer through the analysis are:

- Which features of the listing affect the price most? 
- When is the price highest / lowest?
- What is the volatility of the price?

## Results
Here are the results of the analysis

- <b>Important factors for price.</b> Out of 10 most important factors, many of them are quite intuitive, such as the size and location of the apartment. On top of these factors, availability for the next 60 / 90 days as well as when the calendar is most recently updated are also important.
- <b>The most/least expensive time to stay in D.C. </b>Around mid February seems to be the cheapest time to stay in D.C. On the other hand, Thanksgiving holidays and mid December seem to be most expensive time to stay in D.C. with more than $30 above average.
- <b>Price volatility.</b> The price is very volatile between end of November and early January. The price is quite stable after June.

Full article can be found at the medium post below:

https://medium.com/@yukiteb2/analyzing-the-u-s-capital-through-airbnb-data-eb3e23f1c0ca

