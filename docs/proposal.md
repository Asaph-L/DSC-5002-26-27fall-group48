# Exploring Factors Associated with Used Vehicle Listing Prices on Craigslist

**Project Proposal · SDSC 5002 team project**

## Introduction

Used vehicle listing prices can vary substantially with a vehicle’s age, mileage, condition, manufacturer, and location. Exploring these differences can help describe how sellers price vehicles and which characteristics tend to occur alongside higher or lower asking prices. This project will examine Craigslist vehicle listings from across the United States using exploratory data analysis (EDA) and visualisation.

The analysis will focus on associations in advertised prices. The dataset records sellers’ asking prices rather than completed transactions, so results will not be interpreted as actual sale prices or as evidence that a vehicle characteristic causes a price change.

## Research Questions

1. How are vehicle listing prices distributed, and what unusually low or high values are present?
2. How are vehicle age and odometer reading associated with listing price?
3. How do listing price distributions differ across vehicle condition, manufacturer, fuel type, and vehicle type?
4. How do listing prices vary across US states?

## Dataset Description

The project will use the Craigslist Cars and Trucks dataset published by Austin Reese on Kaggle. The commonly reported version contains approximately 426,880 listings and 26 variables. The file includes numerical fields such as price, model year, odometer reading, and geographic coordinates, alongside categorical fields such as manufacturer, condition, fuel, transmission, vehicle type, and state. Listing identifiers, URLs, descriptions, and posting dates are also included. The exact row count, column names, and file version will be confirmed after download.

The dataset is appropriate for these questions because it contains a large collection of US vehicle listings and records several attributes that can be compared with advertised price. Its main limitations are missing or user-entered information, possible duplicate or inaccurate listings, and the fact that Craigslist listings may not represent the entire used-vehicle market. The outcome is an asking price, not a confirmed transaction price.

Dataset source: Austin Reese, [“Craigslist Cars & Trucks Data,” Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).

## EDA and Data Visualisation Plan

The analysis will begin with a review of the dataset structure, variable types, summary statistics, missing values, and duplicate records. Cleaning decisions will be documented so that their effects on the number of observations and on the price distribution are clear.

1. **Data cleaning:** Check duplicate listing IDs, parse numeric and date fields, and inspect implausible values such as non-positive prices, invalid model years, and extreme odometer readings. Investigate extreme observations before excluding them; do not remove high values solely because they are unusual. Summarise missingness by variable and use appropriate available observations for each analysis rather than filling all missing values automatically.
2. **Derived variable:** Where posting dates are usable, calculate vehicle age as listing year minus model year. If dates cannot be used reliably, analyse model year directly and state the limitation.
3. **Summary statistics:** Report counts, mean, median, standard deviation, quartiles, and range for numerical variables. Report counts and percentages for categorical variables. Because price is likely to be skewed, use the median and interquartile range alongside the mean.
4. **Price distribution:** Use a histogram, and where helpful a log-scaled view, to show skewness and extreme values.
5. **Numerical relationships:** Use scatter plots to examine price against odometer reading and vehicle age. Calculate Pearson or Spearman correlations as appropriate, and explain that correlation measures association rather than causation.
6. **Category comparisons:** Use box plots to compare price across condition, fuel type, and vehicle type. For manufacturer comparisons, focus on the most frequently listed manufacturers so that the chart remains interpretable.
7. **Geographic comparison:** Use a bar chart of median listing price by state, including states with a sufficient number of observations and displaying sample sizes to avoid over-interpreting small groups.

Together, these steps will describe the data, reveal patterns and group differences, and identify data quality issues that affect interpretation. Visualisations will be labelled with units and sample sizes where relevant.

## Other Methods

If time permits, an exploratory multiple linear regression may be fitted using selected vehicle attributes to examine their adjusted associations with listing price. The model will be treated as a supplementary analysis, not as proof of causation or a guaranteed price-prediction tool. Any modelling choices, such as transforming price or grouping rare categories, will be reported.

## Expected Outcomes

The project is expected to provide a clear description of Craigslist vehicle asking prices and how they vary with vehicle age, mileage, condition, manufacturer, vehicle type, and state. It may also identify skewness, missing data, outliers, and uneven representation across categories. The final conclusions will be limited to the listings represented in this dataset and will distinguish observed associations from causal claims.
