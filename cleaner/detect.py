def detect_column_types(df):
    type_dict = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype or "float" in dtype:
            type_dict[col] = "numeric"
        elif "datetime" in dtype:
            type_dict[col] = "date"
        else:
            type_dict[col] = "text"
    return type_dict
