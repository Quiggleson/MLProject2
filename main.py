import re
import json
import requests
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

def writeitemIds(url, headers, body):
    itemids = []

    for startIndex in tqdm(range(0,307,7)):
        body["variables"]["startIndex"] = startIndex
        r = requests.post(url=url, headers=headers, json=body)
        res = r.json()
        products = res["data"]["searchModel"]["products"]
        for product in products:
            itemid = product["identifiers"]["itemId"]
            itemids.append(itemid)

    print(f'unique: {len(np.unique(itemids))}')
    df = pd.DataFrame(np.unique(itemids))
    df.to_csv('./itemids.csv')

def itemidstoresults(itemids):
    
    df = pd.DataFrame(columns=["Brand", "itemId", "stars", "review"])

    for itemid in tqdm(itemids['0'][:217]):

        print(f'trying itemid {itemid}')
        script_path = './metadata_req.sh'
        
        # Modify request to use the current id
        with open(script_path, 'r') as file:
            content = file.read()
            itemidindex = content.index("itemId")
            commaindex = content.index(",",itemidindex)
            content = content[:itemidindex] + f'itemId":"{itemid}"' + content[commaindex:]
            
            with open(script_path, 'w') as file:
                file.write(content)

            # Populate metadata_req.json
            subprocess.run(['bash', script_path])

            # Read metadata_res.json
            with open('metadata_res.json', 'r') as file:
                data = json.load(file)

            brandName = data["data"]["product"]["identifiers"]["brandName"]
            totalReviews = data["data"]["product"]["reviews"]["ratingsReviews"]["totalReviews"]
            totalReviews = min(int(totalReviews), 500)

            for startIndex in range(0, totalReviews, 10):
                
                script_path = './review_req.sh'

                # Modify request to use startIndex
                with open(script_path, 'r') as file:
                    content = file.read()
                    startIndexIndex = content.index("startIndex")
                parenindex = content.index("}",startIndexIndex)
                content = content[:startIndexIndex] + f'startIndex":{startIndex}' + content[parenindex:]
            
                with open(script_path, 'w') as file:
                    file.write(content)

                subprocess.run(['bash', script_path])

                with open('review_res.json', 'r') as file:
                    review_data = json.load(file)

                results = review_data["data"]["reviews"]["Results"]
                
                for result in results:
                    rating = result["Rating"]
                    review = result["ReviewText"]

                    row = (brandName, itemid, rating, review)
                    df.loc[len(df)] = row

    df.to_excel('./data.xlsx', index=False)

def ratingbarchart(df):
    # Count the occurrences of each rating
    rating_counts = df['stars'].value_counts().sort_index()

    # Plotting the bar chart
    plt.bar(rating_counts.index, rating_counts.values, color='skyblue')

    # Adding labels and title
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Ratings')

    # Show the plot
    plt.show()

def brandbarchart(df):
    # Count the occurrences of each rating
    rating_counts = df['Brand'].value_counts().sort_values(ascending=False)

    # Plotting the bar chart
    plt.bar(rating_counts.index, rating_counts.values, color='skyblue')

    # Adding labels and title
    plt.xlabel('Brand')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Brands')
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()

def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        text = re.sub(r'\W', ' ', text)  # Remove non-word characters
        text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with a single space
        text = text.lower()  # Convert to lowercase
        words = text.split()
        words = [ps.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
        return ' '.join(words)

def plot_most_used_words(df, title, color='blue', top_n=10):
    # Tokenize and count words
    words = ' '.join(df['review']).split()
    word_counts = Counter(words)

    # Get the top N words
    top_words = dict(word_counts.most_common(top_n))

    # Plot bar chart
    plt.bar(top_words.keys(), top_words.values(), color=color)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Most Used Words in {title}')
    plt.xticks(rotation=45)
    plt.show()

def cleaning(df, sortreviews = False):

    # Remove nulls
    df_cleaned = df.dropna(subset=['Brand', 'itemId', 'stars', 'review'])
    
    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    # Remove reviews with fewer than X characters
    # Unnecessary, all reviews are of valid length

    if sortreviews:
        # Print df sorted by review length
        df['review_length'] = df['review'].apply(len)

        # Sort the DataFrame by the 'review_length' column
        sorted_df = df.sort_values(by='review_length', ascending=True)

        # Print the sorted DataFrame
        print(sorted_df)

    return df_cleaned

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)['compound']
    return 'positive' if sentiment_score >= 0 else 'negative'

def plot_word_frequency(words, title, ax):
    # Plotting the bar chart using the provided axes
    ax.bar(words.keys(), words.values(), color='skyblue')
    
    # Adding labels and title
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

# Modify the analyze_and_plot_brand_sentiment function to pass the axes to plot_word_frequency
def analyze_and_plot_brand_sentiment(df, brand_column='Brand', review_column='review'):
    # Group by brand
    grouped_by_brand = df.groupby(brand_column)

    # Set up the subplot
    fig, axes = plt.subplots(len(grouped_by_brand), 2, figsize=(12, 6 * len(grouped_by_brand)))

    for i, (brand, group) in enumerate(grouped_by_brand):
        # Separate positive and negative reviews
        positive_reviews = group[group[review_column].apply(analyze_sentiment) == 'positive']
        negative_reviews = group[group[review_column].apply(analyze_sentiment) == 'negative']

        # Tokenize and count words for positive reviews
        positive_words = word_tokenize(preprocess_text(' '.join(positive_reviews[review_column].tolist())))
        positive_word_counts = Counter(positive_words)

        # Tokenize and count words for negative reviews
        negative_words = word_tokenize(preprocess_text(' '.join(negative_reviews[review_column].tolist())))
        negative_word_counts = Counter(negative_words)

        # Get top 10 words for each sentiment
        top_positive_words = dict(sorted(positive_word_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        top_negative_words = dict(sorted(negative_word_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Plot bar charts for positive and negative reviews in the correct subplots
        plot_word_frequency(top_positive_words, f'{len(positive_reviews)} reviews of {brand} (Positive)', axes[i, 0])
        plot_word_frequency(top_negative_words, f'{len(negative_reviews)} reviews of {brand} (Negative)', axes[i, 1])

        # Position the plots in the correct subplots
        axes[i, 0].set_title(f'{brand} (Positive)')
        axes[i, 1].set_title(f'{brand} (Negative)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()

def reviewbased(df, brand_column="Brand", review_column="review", rating_column="stars"):
    # Group by brand
    grouped_by_brand = df.groupby(brand_column)

    # Set up the subplot
    fig, axes = plt.subplots(len(grouped_by_brand), 2, figsize=(12, 6 * len(grouped_by_brand)))

    for i, (brand, group) in enumerate(grouped_by_brand):
        # Separate positive and negative reviews
        positive_reviews = group[group[rating_column].isin([4, 5])]
        negative_reviews = group[group[rating_column].isin([1, 2])]

        # Tokenize and count words for positive reviews
        positive_words = word_tokenize(preprocess_text(' '.join(positive_reviews[review_column].tolist())))
        positive_word_counts = Counter(positive_words)

        # Tokenize and count words for negative reviews
        negative_words = word_tokenize(preprocess_text(' '.join(negative_reviews[review_column].tolist())))
        negative_word_counts = Counter(negative_words)

        # Get top 10 words for each sentiment
        top_positive_words = dict(sorted(positive_word_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        top_negative_words = dict(sorted(negative_word_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Plot bar charts for positive and negative reviews in the correct subplots
        plot_word_frequency(top_positive_words, f'{len(positive_reviews)} reviews of {brand} (Positive)', axes[i, 0])
        plot_word_frequency(top_negative_words, f'{len(negative_reviews)} reviews of {brand} (Negative)', axes[i, 1])

        # Position the plots in the correct subplots
        axes[i, 0].set_title(f'{brand} (Positive)')
        axes[i, 1].set_title(f'{brand} (Negative)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()

url = "https://www.homedepot.com/federation-gateway/graphql?opname=dpdSearchModel"
headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "content-type": "application/json",
        "X-Experience-Name": "major-appliances",
        "apollographql-client-name": "major-appliances",
        "apollographql-client-version": "0.0.0",
        "X-current-url": "/b/Appliances-Refrigerators/N-5yc1vZc3pi",
        "x-hd-dc": "origin",
        "X-Api-Cookies": "{\"x-user-id\":\"a0106286-cdba-d0f1-260f-4641d816e4f4\"}",
        "x-debug": "false",
        "x-thd-customer-token": "",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Referer": "https://www.homedepot.com/b/Appliances-Refrigerators/N-5yc1vZc3pi"
    }
body = {"operationName":"dpdSearchModel","variables":{"skipInstallServices":False,"channel":"DESKTOP","storefilter":"ALL","additionalSearchParams":{"callback":"{\"field\":\"TOP_RATED\",\"order\":\"DESC\"}"},"keyword":"","navParam":"10000003+10000003+564678+7+4294838595","orderBy":{"field":"TOP_RATED","order":"DESC"},"pageSize":7,"startIndex":0,"storeId":"121","zipCode":"48307"},"query":"query dpdSearchModel($storeId: String, $zipCode: String, $skipInstallServices: Boolean = true, $pageSize: Int, $startIndex: Int, $orderBy: ProductSort, $channel: Channel = DESKTOP, $navParam: String, $keyword: String, $itemIds: [String], $storefilter: StoreFilter = ALL, $additionalSearchParams: AdditionalParams) {\n  searchModel(channel: $channel, navParam: $navParam, keyword: $keyword, itemIds: $itemIds, storeId: $storeId, storefilter: $storefilter, additionalSearchParams: $additionalSearchParams) {\n    id\n    products(pageSize: $pageSize, startIndex: $startIndex, orderBy: $orderBy) {\n      itemId\n      dataSources\n      info {\n        hidePrice\n        classNumber\n        hasSubscription\n        isLiveGoodsProduct\n        productDepartment\n        productSubType {\n          name\n          link\n          __typename\n        }\n        globalCustomConfigurator {\n          customExperience\n          __typename\n        }\n        isGenericProduct\n        ecoRebate\n        quantityLimit\n        sskMin\n        sskMax\n        unitOfMeasureCoverage\n        wasMaxPriceRange\n        wasMinPriceRange\n        customerSignal {\n          previouslyPurchased\n          __typename\n        }\n        isBuryProduct\n        returnable\n        isSponsored\n        sponsoredMetadata {\n          campaignId\n          placementId\n          slotId\n          sponsoredId\n          trackSource\n          __typename\n        }\n        augmentedReality\n        sponsoredBeacon {\n          onClickBeacon\n          onViewBeacon\n          onClickBeacons\n          onViewBeacons\n          __typename\n        }\n        samplesAvailable\n        swatches {\n          isSelected\n          itemId\n          label\n          swatchImgUrl\n          url\n          value\n          __typename\n        }\n        totalNumberOfOptions\n        __typename\n      }\n      identifiers {\n        itemId\n        brandName\n        productType\n        canonicalUrl\n        specialOrderSku\n        storeSkuNumber\n        productLabel\n        modelNumber\n        parentId\n        __typename\n      }\n      fulfillment(storeId: $storeId, zipCode: $zipCode) {\n        fulfillmentOptions {\n          type\n          fulfillable\n          services {\n            type\n            locations {\n              inventory {\n                isInStock\n                isOutOfStock\n                isLimitedQuantity\n                isUnavailable\n                quantity\n                maxAllowedBopisQty\n                minAllowedBopisQty\n                __typename\n              }\n              isAnchor\n              curbsidePickupFlag\n              isBuyInStoreCheckNearBy\n              distance\n              locationId\n              state\n              storeName\n              storePhone\n              type\n              __typename\n            }\n            deliveryTimeline\n            deliveryDates {\n              startDate\n              endDate\n              __typename\n            }\n            deliveryCharge\n            dynamicEta {\n              hours\n              minutes\n              __typename\n            }\n            hasFreeShipping\n            freeDeliveryThreshold\n            totalCharge\n            __typename\n          }\n          __typename\n        }\n        anchorStoreStatus\n        anchorStoreStatusType\n        backordered\n        backorderedShipDate\n        bossExcludedShipStates\n        excludedShipStates\n        seasonStatusEligible\n        onlineStoreStatus\n        onlineStoreStatusType\n        __typename\n      }\n      availabilityType {\n        type\n        discontinued\n        buyable\n        status\n        __typename\n      }\n      media {\n        images {\n          url\n          sizes\n          type\n          subType\n          __typename\n        }\n        __typename\n      }\n      pricing(storeId: $storeId) {\n        value\n        original\n        alternatePriceDisplay\n        alternate {\n          bulk {\n            pricePerUnit\n            thresholdQuantity\n            value\n            __typename\n          }\n          unit {\n            caseUnitOfMeasure\n            unitsOriginalPrice\n            unitsPerCase\n            value\n            __typename\n          }\n          __typename\n        }\n        mapAboveOriginalPrice\n        message\n        preferredPriceFlag\n        promotion {\n          type\n          description {\n            shortDesc\n            longDesc\n            __typename\n          }\n          dollarOff\n          percentageOff\n          promotionTag\n          savingsCenter\n          savingsCenterPromos\n          specialBuySavings\n          specialBuyDollarOff\n          specialBuyPercentageOff\n          __typename\n        }\n        specialBuy\n        unitOfMeasure\n        __typename\n      }\n      favoriteDetail {\n        count\n        __typename\n      }\n      installServices(storeId: $storeId, zipCode: $zipCode) @skip(if: $skipInstallServices) {\n        scheduleAMeasure @skip(if: $skipInstallServices)\n        gccCarpetDesignAndOrderEligible @skip(if: $skipInstallServices)\n        __typename\n      }\n      details {\n        installation {\n          serviceType\n          __typename\n        }\n        collection {\n          name\n          url\n          __typename\n        }\n        __typename\n      }\n      badges(storeId: $storeId) {\n        name\n        label\n        __typename\n      }\n      dataSource\n      reviews {\n        ratingsReviews {\n          averageRating\n          totalReviews\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    searchReport {\n      totalProducts\n      sortBy\n      __typename\n    }\n    metadata {\n      canonicalUrl\n      __typename\n    }\n    taxonomy {\n      breadCrumbs {\n        dimensionName\n        label\n        refinementKey\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n}\n"}

r = requests.post(url=url, headers=headers, json=body)
# print(r.json())

# writeitemIds(url, headers, body)

# df = pd.read_csv('./itemids.csv', index_col=0)
# itemidstoresults(df)

df = pd.read_excel('./data.xlsx', header=0)

print(f'size of df: {df}')

# ratingbarchart(df)
# brandbarchart(df)
# process(df)

# Uncomment these if it's the first download
# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt)

# accuracy = process_sentiment_and_plot(df)

df2 = cleaning(df)
print(f'df leng: {len(df)}\ndf2 len: {len(df2)}')

df = df2
# df['review'] = df['review'].apply(preprocess_text)

# brand_subset = ["GE", "Whirlpool", "LG", "Cafe", "KitchenAid"]
brand_subset = ["GE", "Whirlpool", "LG"]
# Selecting a subset of the DataFrame with just the first 2 brands
subset_df = df[df['Brand'].isin(brand_subset)]

# Test the function with the subset DataFrame
# analyze_and_plot_brand_sentiment(subset_df)

# reviewbased(subset_df)