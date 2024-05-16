import pandas as pd

data = pd.read_csv("thanksgiving.csv", encoding="Latin-1")
data.head()

data.columns

data["Do you celebrate Thanksgiving?"].value_counts()

data = data[data["Do you celebrate Thanksgiving?"] == "Yes"]

data["What is typically the main dish at your Thanksgiving dinner?"].value_counts()

data[data["What is typically the main dish at your Thanksgiving dinner?"] == "Tofurkey"]["Do you typically have gravy?"]

data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple"].value_counts()

ate_pies = (pd.isnull(data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple"])
&
pd.isnull(data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pecan"])
 &
 pd.isnull(data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pumpkin"])
)

ate_pies.value_counts()

data["Age"].value_counts()

def extract_age(age_str):
    if pd.isnull(age_str):
        return None
    age_str = age_str.split(" ")[0]
    age_str = age_str.replace("+", "")
    return int(age_str)

data["int_age"] = data["Age"].apply(extract_age)
data["int_age"].describe()

data["How much total combined money did all members of your HOUSEHOLD earn last year?"].value_counts()

def extract_income(income_str):
    if pd.isnull(income_str):
        return None
    income_str = income_str.split(" ")[0]
    if income_str == "Prefer":
        return None
    income_str = income_str.replace(",", "")
    income_str = income_str.replace("$", "")
    return int(income_str)

data["int_income"] = data["How much total combined money did all members of your HOUSEHOLD earn last year?"].apply(extract_income)
data["int_income"].describe()

data[data["int_income"] < 50000]["How far will you travel for Thanksgiving?"].value_counts()

data[data["int_income"] > 150000]["How far will you travel for Thanksgiving?"].value_counts()

data.pivot_table(
    index="Have you ever tried to meet up with hometown friends on Thanksgiving night?", 
    columns='Have you ever attended a "Friendsgiving?"',
    values="int_age"
)

data.pivot_table(
    index="Have you ever tried to meet up with hometown friends on Thanksgiving night?", 
    columns='Have you ever attended a "Friendsgiving?"',
    values="int_income"
)

