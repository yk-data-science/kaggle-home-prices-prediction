import pandas as pd

def engineer_features(df):
    df = df.copy()
    df["price_per_sqft"] = df.get("SalePrice", 0) / (df["GrLivArea"] + 1)
    df["age"] = df["YrSold"] - df["YearBuilt"]
    df["has_basement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["is_new"] = (df["YearBuilt"] == df["YrSold"]).astype(int)
    df["room_to_bath_ratio"] = df["TotRmsAbvGrd"] / (df["FullBath"] + df["HalfBath"] + 1)
    df["garage_score"] = df["GarageArea"] * df["GarageCars"]
    df["overall_score"] = df["OverallQual"] * df["OverallCond"]
    df["age_bin"] = pd.cut(df["age"], bins=[0, 10, 30, 200], labels=["new", "middle", "old"])
    return df