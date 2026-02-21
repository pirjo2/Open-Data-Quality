import json

def make_arrow_safe(df):
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
    return df