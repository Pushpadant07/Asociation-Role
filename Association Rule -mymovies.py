import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules

# As the file is in transaction data we will be reading data directly 
mymovies=pd.read_csv("D:\\ExcelR Data\\Assignments\\Asociation Role\\mymovies.csv")

movies=mymovies.iloc[:,5:]

#going for pairplot for every variables inside my dataset
sns.pairplot(mymovies.iloc[:,0:9])


frequent_movies = apriori(movies,min_support=0.009, max_len=3,use_colnames = True) #in this code im taking support as 0.009
frequent_movies.shape


# Most Frequent item sets based on support 
frequent_movies.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_movies.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_movies.itemsets[1:11])


rules = association_rules(frequent_movies, metric="lift", min_threshold=2) #in this im taking Threshold as 5
rules.shape

# To eliminate Redudancy in Rules
def to_list(i):
    return (sorted(list(i)))


ma_x = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_x  = ma_x .apply(sorted)
rules_sets = list(ma_x )
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
r=rules_no_redudancy.sort_values('lift',ascending=False).head(10)
