import pandas as pd
pd.options.display.max_columns = 99
first_five = pd.read_csv('loans_2007.csv', nrows=5)
first_five

thousand_chunk = pd.read_csv('loans_2007.csv', nrows=1000)
thousand_chunk.memory_usage(deep=True).sum()/(1024*1024)

chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
for chunk in chunk_iter:
    print(chunk.memory_usage(deep=True).sum()/(1024*1024))

chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
total_rows = 0
for chunk in chunk_iter:
    total_rows += len(chunk)
print(total_rows)

# Numeric columns
chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
for chunk in chunk_iter:
    print(chunk.dtypes.value_counts())

# Are string columns consistent across chunks?
obj_cols = []
chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)

for chunk in chunk_iter:
    chunk_obj_cols = chunk.select_dtypes(include=['object']).columns.tolist()
    if len(obj_cols) > 0:
        is_same = obj_cols == chunk_obj_cols
        if not is_same:
            print("overall obj cols:", obj_cols, "\n")
            print("chunk obj cols:", chunk_obj_cols, "\n")    
    else:
        obj_cols = chunk_obj_cols

## Create dictionary (key: column, value: list of Series objects representing each chunk's value counts)
chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
str_cols_vc = {}
for chunk in chunk_iter:
    str_cols = chunk.select_dtypes(include=['object'])
    for col in str_cols.columns:
        current_col_vc = str_cols[col].value_counts()
        if col in str_cols_vc:
            str_cols_vc[col].append(current_col_vc)
        else:
            str_cols_vc[col] = [current_col_vc]

## Combine the value counts.
combined_vcs = {}

for col in str_cols_vc:
    combined_vc = pd.concat(str_cols_vc[col])
    final_vc = combined_vc.groupby(combined_vc.index).sum()
    combined_vcs[col] = final_vc

combined_vcs.keys()

obj_cols

useful_obj_cols = ['term', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'issue_d', 'purpose', 'earliest_cr_line', 'revol_util', 'last_pymnt_d', 'last_credit_pull_d']

for col in useful_obj_cols:
    print(col)
    print(combined_vcs[col])
    print("-----------")

convert_col_dtypes = {
    "sub_grade": "category", "home_ownership": "category", 
    "verification_status": "category", "purpose": "category"
}

chunk[useful_obj_cols]

chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000, dtype=convert_col_dtypes, parse_dates=["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"])

for chunk in chunk_iter:
    term_cleaned = chunk['term'].str.lstrip(" ").str.rstrip(" months")
    revol_cleaned = chunk['revol_util'].str.rstrip("%")
    chunk['term'] = pd.to_numeric(term_cleaned)
    chunk['revol_util'] = pd.to_numeric(revol_cleaned)
    
chunk.dtypes

chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000, dtype=convert_col_dtypes, parse_dates=["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"])
mv_counts = {}
for chunk in chunk_iter:
    term_cleaned = chunk['term'].str.lstrip(" ").str.rstrip(" months")
    revol_cleaned = chunk['revol_util'].str.rstrip("%")
    chunk['term'] = pd.to_numeric(term_cleaned)
    chunk['revol_util'] = pd.to_numeric(revol_cleaned)
    float_cols = chunk.select_dtypes(include=['float'])
    for col in float_cols.columns:
        missing_values = len(chunk) - chunk[col].count()
        if col in mv_counts:
            mv_counts[col] = mv_counts[col] + missing_values
        else:
            mv_counts[col] = missing_values
mv_counts

chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000, dtype=convert_col_dtypes, parse_dates=["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"])
mv_counts = {}
for chunk in chunk_iter:
    term_cleaned = chunk['term'].str.lstrip(" ").str.rstrip(" months")
    revol_cleaned = chunk['revol_util'].str.rstrip("%")
    chunk['term'] = pd.to_numeric(term_cleaned)
    chunk['revol_util'] = pd.to_numeric(revol_cleaned)
    chunk = chunk.dropna(how='all')
    float_cols = chunk.select_dtypes(include=['float'])
    for col in float_cols.columns:
        missing_values = len(chunk) - chunk[col].count()
        if col in mv_counts:
            mv_counts[col] = mv_counts[col] + missing_values
        else:
            mv_counts[col] = missing_values
mv_counts

