import json
import pandas as pd


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()

    for col in safe_df.columns:
        s = safe_df[col]

        def _fix_value(x):
            if x is None:
                return None

            try:
                if pd.isna(x):
                    return None
            except Exception:
                pass

            if isinstance(x, (list, dict, tuple)):
                return json.dumps(x, ensure_ascii=False)

            if isinstance(x, (bytes, bytearray)):
                try:
                    return x.decode("utf-8", errors="replace")
                except Exception:
                    return str(x)

            return x

        s = s.map(_fix_value)

        non_null = s.dropna()
        if non_null.empty:
            safe_df[col] = s
            continue

        observed_types = {type(x) for x in non_null}

        if len(observed_types) > 1:
            safe_df[col] = s.map(lambda x: None if x is None else str(x))
        else:
            safe_df[col] = s

    return safe_df