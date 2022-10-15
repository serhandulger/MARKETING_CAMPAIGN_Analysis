import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
#import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

import pandas as pd
df=pd.read_csv('/Users/serhandulger/Desktop/marketing.csv')

# First look to dataset

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### NA SUM #####################")
    print(dataframe.isnull().sum().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())
    print("##################### Nunique #####################")
    print(dataframe.nunique())

check_df(df)

df["converted"] = df["converted"].astype("object")
df["is_retained"] = df["is_retained"].astype("object")

# Examining dataset in terms of the determine NULL values

def data_visualizations(data):
    import missingno as msno
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("VISUALIZING", "\n\n")
    msno.bar(data)
    plt.show()
    #msno.heatmap(data)
    #plt.show()
    msno.matrix(data)
    plt.show()
    #print("CORRELATION GRAPH", "\n\n")
    #plt.figure(figsize=(14, 12))
    #sns.heatmap(data.corr(), annot=True, cmap="BuPu")
    #plt.show()
    #sns.pairplot(data)
    #plt.show()


# To be able to better understanding of dataset, it is a function to grab categorical - numerical and cardinal variables

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

grab_col_names(df)

df.dtypes

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    import seaborn as sns
    import matplotlib.pyplot as plt
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

cat_cols , num_cols, cat_but_car = grab_col_names(df)

cat_cols

bool_error = ["converted","is_retained","DoW"]

cat_cols = [col for col in cat_cols if col not in bool_error]

cat_cols

for i in cat_cols:
    print(cat_summary(df,i,plot=True))

# Data Preparation for Analysis & Exploratory Data Analysis

channel_dict = {"House Ads": 1, "Instagram":2, "Facebook":3,"Email":4,"Push":5}

df["channel_coding"] = df["subscribing_channel"].map(channel_dict)

df["is_correct_language"] = np.where(df["language_preferred"] == df["language_displayed"], "Yes","No")

import datetime as dt
df["date_subscribed"] = pd.to_datetime(df["date_subscribed"]).dt.normalize()
df["date_served"] = pd.to_datetime(df["date_served"]).dt.normalize()
df["date_canceled"] = pd.to_datetime(df["date_canceled"]).dt.normalize()

def create_date_features(df):
    df['subscribed_month'] = df["date_subscribed"].dt.month
    df['subscribed_day_of_month'] = df["date_subscribed"].dt.day
    df['subscribed_day_of_year'] = df["date_subscribed"].dt.dayofyear
    df['subscribed_week_of_year'] = df["date_subscribed"].dt.weekofyear
    df['subscribed_day_of_week'] = df["date_subscribed"].dt.dayofweek
    df['subscribed_year'] = df["date_subscribed"].dt.year
    df["subscribed_is_wknd"] = df["date_subscribed"].dt.weekday // 4
    df['subscribed_is_month_start'] = df["date_subscribed"].dt.is_month_start.astype(int)
    df['subscribed_is_month_end'] = df["date_subscribed"].dt.is_month_end.astype(int)
    return df

# Arranging date variables using lambda function (alternative way)
df["subscribed_year_2"] = df["date_subscribed"].apply(lambda x: x.year)
df["subscribed_month_2"] = df["date_subscribed"].apply(lambda x: x.month)

create_date_features(df)

# Daily users finding
daily_users = df.groupby(["date_served"])["user_id"].nunique()

daily_users

# Average daily users
daily_users.mean()

# How many times subscriptons was made by each user
n_orders = df.groupby(["user_id"])["date_subscribed"].nunique().sort_values()
n_orders

import numpy as np
mult_orders_perc = np.sum(n_orders > 1) / df["user_id"].nunique()
print(f"{100* mult_orders_perc: .2f}% customer has made subscription more than 1 time")

import seaborn as sns
ax = sns.distplot(n_orders, kde=False, hist=True)
ax.set(title="Subscription distribution per customer",
      xlabel="Subscription amount",
      ylabel="Customer amount")

# Which age group cancels subsription in average day ?
df["subs_and_cancel_timing_difference"] = df["date_canceled"] - df["date_subscribed"]

analysis = df.groupby(["age_group","marketing_channel"])["subs_and_cancel_timing_difference"].mean().to_frame("Mean").reset_index()

analysis["days"] = analysis["Mean"].apply(lambda x: x.days)

analysis.head()

sns.barplot(x="marketing_channel",y="days",hue="age_group",data=analysis)

import matplotlib.pyplot as plt
daily_users.plot()
plt.title("Daily number of users who see ads")
plt.xlabel("Date")
plt.ylabel("Number of Users")
plt.xticks(rotation=45)
plt.show()

# We checked the dataset to see if there are any marketing channels in other years.
sns.catplot(x="marketing_channel",y="subscribed_year",data=df)

sns.barplot(x="marketing_channel",y="subscribed_day_of_month",hue="language_displayed",data=df)

grouping = df.groupby(["marketing_channel","subscribed_day_of_month"])["user_id"].nunique().to_frame(name="Count").reset_index()

grouping

grouping_arranged = grouping.pivot("marketing_channel","subscribed_day_of_month","Count")

grouping_arranged

plt.figure(figsize=(15,10))
sns.set(font_scale=1)
sns.heatmap(grouping_arranged, annot=True, linewidths=.10 , fmt = ".2g")

grouping2 = df.groupby(["age_group","subscribed_day_of_month"])["user_id"].nunique().to_frame(name="Count").reset_index()

grouping_arranged2 = grouping2.pivot("age_group","subscribed_day_of_month","Count")

grouping_arranged2

plt.figure(figsize=(15,10))
sns.set(font_scale=1)
sns.heatmap(grouping_arranged2, annot=True, linewidths=.10 , fmt = ".2g")

# Introduction to Common Marketing Metrics

## Conversion Rate = #of People who convert / Total #of people we marketed to

subscribers = df[df["converted"] == True]["user_id"].nunique()

subscribers

total = df["user_id"].nunique()

conversion_rate = subscribers / total

print(round(conversion_rate*100,2),"%")

## Retention Rate

## Retention Rate =  #of People who remained subscribed / Total #of people who converted

df["is_retained"].isnull().sum()

df["is_retained"].mode()[0]

df["is_retained"].fillna(df["is_retained"].mode()[0],inplace=True)

df["is_retained"].isnull().sum()

retained = df[df["is_retained"] == True]["user_id"].nunique()

subscribers = df[df["converted"] == True]["user_id"].nunique()

print( f" Retained customer", retained)
print( f" Total Subscribers", subscribers)

retention = retained / subscribers

#RETENTION RATE
print(round(retention*100,2),"%")

# Basic Customer Segmentation by using Pandas

house_ads = df[df["subscribing_channel"] == "House Ads"]

retained_ha = house_ads[house_ads["is_retained"]==True]["user_id"].nunique()
subscribers_ha = house_ads[house_ads["converted"]==True]["user_id"].nunique()

retention_rate = retained_ha / subscribers_ha

#RETENTION RATE
#Number of users retained by the number of subscribers who originally subscribed through a House ADS

print(f" Retention Rate for House Ads Subscribers :", retention_rate*100, "%")

# Segmenting using pandas

# Group by subscribing channel and calculate retention

retained1 = df[df["is_retained"] == True].groupby(["subscribing_channel"])["user_id"].nunique()

# Group by subscribing channel and calculating subscribers

subscribers1 = df[df["converted"] == True].groupby(["subscribing_channel"])["user_id"].nunique()

print( f" Retained customer", retained1)
print( f" Total Subscribers", subscribers1)

## Segmenting Results

#Calculate the retention rate accross the DataFrame

channel_retention_rate = (retained1/subscribers1)*100
print(channel_retention_rate)

#Comparing language conversion rate
# Group by language_displayed and count unique users

total = df.groupby(df["language_displayed"])["user_id"].nunique()

total

# Group by language displayed and count unique conversions

subscribers3 = df[df["converted"]==True].groupby(["language_displayed"])["user_id"].nunique()

subscribers3

# Calculate the conversion rate for all languages

language_conversion_rate = subscribers3/total

print(f" Language conversion rates for each languages :", language_conversion_rate*100, "%")

## Aggregating by date
# Group by date_served and count unique users

total = df.groupby(["date_served"])["user_id"].nunique()
print(total)

#Group by date_served and count unique converted users
subscribers4 = df[df["converted"]==True].groupby(df["date_served"])["user_id"].nunique()

# Calculate the conversion rate per day
daily_conversion_rate = subscribers4/total

print(daily_conversion_rate)

# Plotting Campaign Results

# Comparing language conversion rate

import matplotlib.pyplot as plt

# Creating a barchart using channel retention dataframe

language_conversion_rate.plot(kind="bar")
# Add a title and x and y-axis labels
plt.title('Conversion rate by language\n', size = 16)
plt.ylabel('Conversion rate (%)', size = 14)
plt.xlabel('Language', size = 14)

# Display the plot
plt.show()

## Reset index to turn the Series into a DataFrame

daily_conversion_rate = pd.DataFrame(daily_conversion_rate.reset_index())

daily_conversion_rate

## Rename the columns

daily_conversion_rate.columns = ["date_served","conversion_rate"]

daily_conversion_rate.plot("date_served","conversion_rate")

plt.title('Daily conversion rate\n', size = 16)
plt.ylabel('Conversion rate (%)', size = 14)
plt.xlabel('Date', size = 14)

# Set the y-axis to begin at 0
plt.ylim(0)

# Display the plot
plt.show()

# Marketing channels across age groups

df.groupby("marketing_channel")["age_group"].count()

channel_age = df.groupby(["marketing_channel","age_group"])["user_id"].count()
channel_age

# Unstack channel age and transform it into a DataFrame

channel_age_df = pd.DataFrame(channel_age.unstack(level=1))

channel_age_df

# Plot the results

channel_age_df.plot(kind="bar")

plt.title('Marketing channels by age group')
plt.xlabel('Age Group')
plt.ylabel('Users')
# Add a legend to the plot
plt.legend(loc = 'upper right',
           labels = channel_age_df.columns.values)
plt.show()

# Count the subs by subscribing channel and date subscribed

retention_total = df.groupby(["date_subscribed","subscribing_channel"])["user_id"].nunique()

print(retention_total)

# Count the retained subs by subscribing channel and date subscribed

retention_subs = df[df["is_retained"] == True].groupby(["date_subscribed","subscribing_channel"])["user_id"].nunique()

# Print results
print(retention_subs)

# Divide retained subscribers by total subscribers

retention_rate = retention_subs / retention_total
retention_rate

retention_rate_df = pd.DataFrame(retention_rate.unstack(level=1))

retention_rate_df

# Plotting retention rate
retention_rate_df.plot()

# Add a title, x-label, y-label, legend and display the plot
plt.title('Retention Rate by Subscribing Channel')
plt.xlabel('Date Subscribed')
plt.ylabel('Retention Rate (%)')
plt.legend(loc = 'upper right',
           labels = retention_rate_df.columns.values)
plt.show()

# Conversion Attribution

# Building functions to automate analysis

def conversion_rate(dataframe, column_names):
    # Total number of converted users
    column_conv = dataframe[dataframe["converted"] == True].groupby(column_names)["user_id"].nunique()

    # Total number of users
    column_total = dataframe.groupby(column_names)["user_id"].nunique()

    # Conversion rate
    conversion_rate = column_conv / column_total

    # Fill missing values with 0
    conversion_rate = conversion_rate.fillna(0)

    return conversion_rate

## Test and visualize conversion function

# Calculate conversion rate by age group

age_group_conv = conversion_rate(df,["date_served","age_group"])

age_group_conv

# Unstack and create a dataframe

age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

# Visualize conversion by age_group

age_group_df.plot()
plt.title('Conversion rate by age group\n', size = 16)
plt.ylabel('Conversion rate', size = 14)
plt.xlabel('Age group', size = 14)
plt.show()

# Scenarios

# We've looked at conversion rate by age, you want to see if that trend has changed over time.
# Marketing has been changing their strategy and wants to make sure that their new method isn't alienating age groups that are less comfortable with their product.
# However, to do so, we need to create a plotting function to make it easier to visualize your results.

def plotting_conv(dataframe):
    for x in dataframe:
        #Plot column by dataframe's index
        plt.plot(dataframe.index,dataframe[x])
        plt.title('Daily ' + str(x) + ' conversion rate\n',
                  size = 16)
        plt.ylabel('Conversion rate', size = 14)
        plt.xlabel('Date', size = 14)
        # Show plot
        #plt.figure(figsize=(44,22))
        plt.show()
        plt.clf()

# Putting it all together
# Your marketing stakeholders have requested a report of the daily conversion rate for each age group, and they need it as soon as possible.
# They want you to refresh this report on a monthly cadence. This is a perfect opportunity to utilize your functions.
# Not only will the functions help you get this report out promptly today, but it will also help each month when it's time for a refresh of the data.
# Remember, conversion_rate() takes a DataFrame and a list of columns to calculate the conversion rate.

# Calculate conversion rate by date served and age group

age_group_conv = conversion_rate(df,["date_served","age_group"])

# Unstack age_group_conv and create a dataframe

age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

#Plot the results

plotting_conv(age_group_df)

# Identifying Inconsistencies

# House ads conversion rate
# The house ads team has become worried about some irregularities they've noticed in conversion rate.
# It is common for stakeholders to come to you with concerns they've noticed around changing metrics.
# As a data scientist, it's your job to determine whether these changes are natural fluctuations or if they require further investigation.
# In this exercise, you'll try out your conversion_rate() and plotting_conv() functions out on marketing looking at conversion rate by 'date_served' and 'marketing_channel'.

# Calculate conversion rate by date served and channel

daily_conv_channel = conversion_rate(df, ["date_served","marketing_channel"])

print(daily_conv_channel)

# Calculate conversion rate by date served and channel

daily_conv_channel = conversion_rate(df, ["date_served","marketing_channel"])

# Unstack daily_conv_channel and convert it to a Dataframe

daily_conv_channel = pd.DataFrame(daily_conv_channel.unstack(level=1))

# Plot the results of daily_conv_channel
plotting_conv(daily_conv_channel)

# Analyzing HouseAds Conversion Rate

# Now that you have confirmed that house ads conversion has been down since January 11, you will try to identify potential causes for the decrease.
# As a data scientist supporting a marketing team, you will run into fluctuating metrics all the time. It's vital to identify if the fluctuations are due to expected shifts in user behavior (i.e., differences across the day of the week) versus a larger problem in technical implementation or marketing strategy.
# In this exercise, we will begin by checking whether users are more likely to convert on weekends compared with weekdays and determine if that could be the cause for the changing house ads conversion rate.

df["DoW_served"] = df["date_served"].dt.dayofweek

# Calculate day of conversion rate by day of week

DoW_conversion = conversion_rate(df,["DoW_served","marketing_channel"])

# Unstack channels

DoW_df = pd.DataFrame(DoW_conversion.unstack(level=1))

# Plot conversion rate by day of week

DoW_df.plot()
plt.title('Conversion rate by day of week\n')
plt.ylim(0)
plt.show()

# House ads conversion by language

# Now that you've ruled out natural fluctuations across the day of the week a user saw our marketing assets as they cause for decreasing house ads conversion, you will take a look at conversion by language over time. Perhaps the new marketing campaign does not apply broadly across different cultures.
# Ideally, the marketing team will consider cultural differences prior to launching a campaign, but sometimes mistakes are made, and it will be your job to identify the cause. Often data scientists are the first line of defense to determine what went wrong with a marketing campaign. It's your job to think creatively to identify the cause.

# Isolate the rows where marketing channel is house ads

house_ads = df[df["marketing_channel"] == "House Ads"]

# Calculate conversion by date served and language displayed

conv_lang_channel = conversion_rate(house_ads, ["date_served","language_displayed"])

# Unstack conv_lang_channel

conv_lang_df = pd.DataFrame(conv_lang_channel.unstack(level=1))

# Use plotting function to display results

plotting_conv(conv_lang_df)

# Creating a DataFrame for house ads

# The house ads team is concerned because they've seen their conversion rate drop suddenly in the past few weeks. In the previous exercises, you confirmed that conversion is down because you noticed a pattern around language preferences.
# As a data scientist, it is your job to provide your marketing stakeholders with as specific feedback as possible as to what went wrong to maximize their ability to correct the problem. It is vital that you not only say "looks like there's a language problem," but instead identify what the problem is specifically so that the team doesn't repeat their mistake.

# Add the new column is correct language

house_ads["is_correct_langg"] = np.where(house_ads["language_preferred"]== house_ads["language_displayed"],"Yes","No")

# Group by date_served and is_correct_langg

language_check = house_ads.groupby(["date_served","is_correct_langg"])["is_correct_langg"].count()

language_check

# Unstack language check and fill missing values with 0's

language_check_df = pd.DataFrame(language_check.unstack(level=1)).fillna(0)

print(language_check_df)

# Confirming house ads error

# Now that you've created a DataFrame that checks whether users see ads in the correct language let's calculate what percentage of users were not being served ads in the right language and plot your results.

# Divide the count where language is correct by the row sum

language_check_df["pct"] = language_check_df["Yes"] / language_check_df.sum(axis=1)

# Plot and show your results

plt.plot(language_check_df.index.values, language_check_df["pct"])
plt.show()

# Setting up conversion indexes

# Now that you've determined that language is, in fact, the issue with House Ads conversion, stakeholders need to know how many subscribers they lost as a result of this bug.
# In this exercise, you will index non-English language conversion rates against English conversion rates in the time period before the language bug arose

# Calculate pre-error conversion rate
house_ads_bug = house_ads[house_ads['date_served'] < '2018-01-11']
lang_conv = conversion_rate(house_ads_bug, ['language_displayed'])

# Index other language conversion rate against English
spanish_index = lang_conv['Spanish']/lang_conv['English']
arabic_index = lang_conv['Arabic']/lang_conv['English']
german_index = lang_conv['German']/lang_conv['English']

print("Spanish index:", spanish_index)
print("Arabic index:", arabic_index)
print("German index:", german_index)

# Analyzing user preferences

# To understand the true impact of the bug, it is crucial to determine how many subscribers we would have expected had there been no language error. This is crucial to understanding the scale of the problem and how important it is to prevent this kind of error in the future.
# In this step, you will create a new DataFrame that you can perform calculations on to determine the expected number of subscribers. This DataFrame will include how many users prefer each language by day. Once you have the DataFrame, you can begin calculating how many subscribers you would have expected to have had the language bug not occurred.

# Group house_ads by date and language

converted = house_ads.groupby(["date_served","language_preferred"]).agg({"user_id":"nunique","converted":"sum"})

# Unstack converted

converted_df = pd.DataFrame(converted.unstack(level=1))

converted_df

# Creating a DataFrame based on indexes
#
# Now that you've created an index to compare English conversion rates against all other languages, you will build out a DataFrame that will estimate what daily conversion rates should have been if users were being served the correct language.
# An expected conversion DataFrame named converted has been created for you grouping house_ads by date and preferred language. It contains a count of unique users as well as the number of conversions for each language, each day.
# For example, you can access the number of Spanish-speaking users who received house ads using converted[('user_id','Spanish')]

# Create English conversin rate column for affected period
converted_df["english_conv_rate"] = converted_df.loc['2018-01-11':'2018-01-31'][("converted","English")]
converted_df

# Create expected conversion rates for each language
converted_df['expected_spanish_rate'] = converted_df['english_conv_rate']*spanish_index
converted_df['expected_arabic_rate'] = converted_df['english_conv_rate']*arabic_index
converted_df['expected_german_rate'] = converted_df['english_conv_rate']*german_index
converted_df

# Multiply number of users by the expected conversion rate
converted_df['expected_spanish_conv'] = converted_df['expected_spanish_rate']/100*converted_df[('user_id','Spanish')]
converted_df['expected_arabic_conv'] = converted_df['expected_arabic_rate']/100*converted_df[('user_id','Arabic')]
converted_df['expected_german_conv'] = converted_df['expected_german_rate']/100*converted_df[('user_id','German')]
converted_df

# Assessing bug impact

# It's time to calculate how many subscribers were lost due to mistakenly serving users English rather than their preferred language. Once the team has an estimate of the impact of this error, they can determine whether it's worth putting additional checks in place to avoid this in the future—you might be thinking, of course, it's worth it to try to prevent errors! In a way, you're right, but every choice a company makes requires work and funding. The more information your team has, the better they will be able to evaluate this trade-off.
# The DataFrame converted has already been loaded for you. It contains expected subscribers columns for Spanish, Arabic and German language speakers named expected_spanish_conv, expected_arabic_conv and expected_german_conv respectively.

# Use .loc to slice only the relevant dates
converted_df = converted_df.loc['2018-01-11':'2018-01-31']

# Sum expected subscribers for each language
expected_subs = converted_df['expected_spanish_conv'].sum() + converted_df['expected_arabic_conv'].sum() + converted_df['expected_german_conv'].sum()

# Calculate how many subscribers we actually got
actual_subs = converted_df[('converted','Spanish')].sum() + converted_df[('converted','Arabic')].sum() + converted_df[('converted','German')].sum()

# Subtract how many subscribers we got despite the bug
lost_subs = expected_subs - actual_subs
print(lost_subs)

# Use .loc to slice only the relevant dates
converted_df = converted_df.loc['2018-01-11':'2018-01-31']

# Sum expected subscribers for each language
expected_subs = converted_df['expected_spanish_conv'].sum() + converted_df['expected_arabic_conv'].sum() + converted_df['expected_german_conv'].sum()

# Calculate how many subscribers we actually got
actual_subs = converted_df[('converted','Spanish')].sum() + converted_df[('converted','Arabic')].sum() + converted_df[('converted','German')].sum()

# Subtract how many subscribers we got despite the bug
lost_subs = expected_subs - actual_subs
print(lost_subs)