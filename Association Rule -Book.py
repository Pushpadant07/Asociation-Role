import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns  
from mlxtend.frequent_patterns import apriori,association_rules

book=pd.read_csv("D:\\ExcelR Data\\Assignments\\Asociation Role\\book.csv")

#going for visualizations
sns.pairplot(book.iloc[:,0:10])

# removing the last empty transaction
books = book.iloc[:1999,:]
#applying the apriori
frequent_itemsets = apriori(book,min_support=0.007, max_len=3,use_colnames = True) #im using support as 0.007
frequent_itemsets.shape

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)

plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2) # Threshold was 1 but now im using 2
rules.shape


rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

 
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
rules_no_redudancy = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
r=rules_no_redudancy.sort_values('lift',ascending=False).head(10)
