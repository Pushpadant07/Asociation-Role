import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori,association_rules

groceries = []
# As the file is in transaction data we will be reading data directly 
with open("D:\\ExcelR Data\\Assignments\\Asociation Role\\groceries.csv","r") as f:
    groceries = f.read()



# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
    
all_groceries_list = []

#for i in groceries_list:
#here im taking out each and every item  from that groceries_list and assign it based on index values
all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter

item_frequencies = Counter(all_groceries_list)
# after sorting
#item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
plt.bar(height = frequencies[:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[:11]);plt.xlabel("items")
plt.ylabel("Count")


# Creating Data Frame for the transactions data 

# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X,min_support=0.009, max_len=3,use_colnames = True) #in this code im taking support as 0.009
frequent_itemsets.shape


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2) #in this im taking Threshold as 5
rules.shape

# Visulize of rules
plt.bar(height = frequencies[:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),rules[:11]);plt.xlabel("rules")
plt.ylabel("Count")

rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True) #sorting lift column ascending order

 
# To eliminate Redudancy in Rules
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
r=rules_no_redudancy.sort_values('lift',ascending=False).head(10)



