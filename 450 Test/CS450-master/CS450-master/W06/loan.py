import pandas as pd

#get_loans
#
#This function prepares the loan dataset (Br. Burton's own unique dataset) for usage
#by the ID3 algorithm
def get_loans():
    headers = [
        "credit score",
        "income",
        "collateral",
        "should_loan"
    ]

    df = pd.read_csv("loan.csv", delimiter=",", header = None, names = headers)

    return df[headers[0:3]], df[headers[3:4]], headers