# Santander-Product-Competition

This repository contains my solution to the Santander Product Competition on Kaggle.
https://www.kaggle.com/jeremy123w


Santander is a banking company in Spain that offers an array of financial products.  Their current system was flawed in that 
a small number of their customers would receive many recommendations while many others would not see any at all.  Santander 
tasked kaggle competitors to accurately predict what their existing customers would use in the next month.

https://www.kaggle.com/c/santander-product-recommendation

I used a single xgboost model that trained on 212 features across approximately 46,000 customer accounts.  Some of the key
aspects of my model was I linked the customer ID across previous months to determine lagged features such as previous products
they have owned.  I also analyzed product distribution across continuous features such as age to intelligently bucketize them.

The data can be downloaded here:
https://www.kaggle.com/c/santander-product-recommendation/data


create_features_and_test_set.py -This script parses the entire training and test sets to create the features and labels
income_by_age_and_nomprov.py- This script is used to map missing incomes to average incomes within select age groups and cities
run_xgboost.py- This script runs a single xgboost model on 95% of the training data with a 5% CV set and then predicts the test set

