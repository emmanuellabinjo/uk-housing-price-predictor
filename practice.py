import pandas as pd

data = {
    "address": ["10 Baker St", "22 Oxford Rd", "5 Park Lane", "8 High St"],
    "city": ["London", "Manchester", "London", "Birmingham"],
    "price": [450000, 210000, 980000, 175000],
    "bedrooms": [2, 3, 4, 2]
}

df = pd.DataFrame(data)
print(df.head())

print(df["price"])
print(df["city"] == "London")
print(df[df["city"] == "London"])
print(df[df["price"] > 300000])

print(df["price"].mean())
print(df["price"].max())
print(df.groupby("city")["price"].mean())

df["price_per_bedroom"] = df["price"] / df["bedrooms"]
print(df.head())