import pandas as pd

def get_voting():
    headers = [
        'party',                #1
        'infant',               #2
        'water',                #3
        'budget-resolution',    #4
        'physician-freeze',     #5
        'salvador-aid',         #6
        'religion-school',      #7
        'anti-satellite',       #8
        'nicaraguan-aid',       #9
        'mx-missile',           #10
        'immigration',          #11
        'synfuels-corporation', #12
        'education',            #13
        'superfund',            #14
        'crime',                #15
        'duty-free-exports',    #16
        'export-south-africa'   #17
    ]

    obj = {
        'democrat': 0,
        'republican': 1,
        'n': 0,
        'y': 1,
        '?': 2
    }


    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",
                        delimiter=",", header=None, names=headers)

    #replace all of the datset values with the values in obj
    df.replace(obj, inplace=True)
    
    #split dataset
    train_set = df[headers[1:]]
    target_set = df[headers[0:1]]
    
    #this is kind of a retarded way to move the first column (targets) to the last column
    #so it would work
    headers.remove('party')
    headers.append('party')

    return train_set, target_set, headers