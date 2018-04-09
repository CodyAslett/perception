
# coding: utf-8

# In[1]:


from carEval import carEval
from indian_diabete import indian_diabete
from automobile_mpg import automobile_mpg
import pandas as pd
def get_k():
    while True:
        try:
            k = int(raw_input("Please enter in the number of neighbor(s): "))
            break
        except ValueError:
            print("Please enter in a valid number")
    return k      


# In[2]:


def get_fold():
    while True:
        try:
            fold = int(raw_input("Please enter in the number of fold(s): "))
            break
        except ValueError:
            print("Please enter in a valid number")
    return fold      
            


# In[3]:


def pretty_print(my_scores, sk_scores,k,fold):
    d = {'Test:': [i for i in xrange(1,fold+1)],'my_score': my_scores, 'sk_score': sk_scores}
    df = pd.DataFrame(data=d)
    
    print "Below is the result of", k, "neighbors and", fold, "test"
    
    print df
    
    print "Average of my scores are:", round(my_scores.mean(),2)
    print "Average of sk scores are:", round(sk_scores.mean(),2)


# In[4]:


def main():
    print "1 for Car Evaluation"
    print "2 for Pima Indian Diabetes"
    print "3 for Automobile MPG"
    print ""
    
    my_input = raw_input("Please select your dataset: ")
    my_input = int(my_input)
    while(True):
        if my_input == 1 or my_input == 2 or my_input == 3:
            break
        my_input = raw_input("Please enter either 1, 2, or 3: ")
        my_input = int(my_input)
        
        
    k = get_k()
    fold = get_fold()
        
    my_scores = []
    sk_scores = []
    
    if my_input == 1:
        my_scores, sk_scores = carEval(k,fold)
        pretty_print(my_scores, sk_scores,k,fold)
    elif my_input == 2:
        my_scores, sk_scores = indian_diabete(k, fold)
        my_scores, sk_scores = carEval(k,fold)
        
        
    elif my_input == 3:
        automobile_mpg(k, fold)
        print "Data has been clearn up"
    


# In[5]:


main()

