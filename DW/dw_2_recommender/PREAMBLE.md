**Task description (legend)**<br/>
<br/>
Hello! In this final project, the company Instacart has come to you for help in building their recommendation system. Below is a letter in which they describe what they want.<br/>
<br/>
Regardless of whether you buy spontaneously or plan your purchases carefully, your unique consumer behavior determines who you are. Instacart is an application for ordering and delivering groceries. It helps simplify the process of filling your fridge with your favorite products when you need them. After selecting items through the Instacart app, our employees review the order, make purchases, and deliver them from the store to the customer's home.<br/>
<br/>
Our Data Science team plays a huge role in organizing this user experience. Currently, we use transactional data to develop models that predict which products the user will buy again, try for the first time, or add to their cart during the session.<br/>
<br/>
The list of goods and products is enormous, and it can be difficult to search for something in it. Therefore, we want to help the user by showing them the products that they are most likely to want to buy. Use anonymous data about customer orders to predict which products they will order next time. We will display your predictions to customers on the main page to make the service more convenient.<br/>
<br/>
And if you are not only very good at creating models, but also able to deploy them in production, we have an additional request. Wrap the developed algorithm in a Python library that our backend developers can use in the future.<br/>
<br/>
**Technical task for the final project**<br/>
 - The final project is a competition on the Kaggle platform. [Link](https://www.kaggle.com/competitions/skillbox-recommender-system/overview).
 - You need to write reproducible code that generates a file with 10 items for each customer that they are most likely to purchase in the next transaction.
 - You can submit no more than five solutions for validation per day. In the Leaderboard tab, you can find two basic solutions:
     - The top 10 popular products for all customers (the same as in the sample_submission.csv file).
     - A benchmark solution that you need to beat to achieve a score of no less than satisfactory.
 - You can find the data description, file format for the response, and quality metric in the competition on Kaggle.

 **Requirements**<br/>
<br/>
The project does not have strict requirements for the algorithms to be used in its implementation. Hybrid recommender systems and approaches/algorithms covered during the theoretical classes are encouraged to be used.<br/>
<br/>

**Enhancements and Additions to the Task**<br/>
<br/>
The system of generating a file with products that will be in the customer's next purchase is not always suitable for production output. Fulfill the client's additional request: wrap the developed recommendation system into a Python class with the following methods:
- Provide a set of K most relevant products for a user by user ID.
- Provide an array of sets of K most relevant products for users by an array of user IDs.
- Add fresh transaction data.
- Update/add data on product characteristics.
- Retrain the recommendation system.<br/>

When implementing the class, ensure the efficiency of the method's speed and memory usage. Try to use parallel computing where possible. The target model should be trained in no more than five hours and make predictions for all customers in no more than 15 minutes.

**The delivery format and evaluation are as follows:**<br/>
<br/>
The first step is to ensure that the solution on Kaggle receives a score higher than the "Pass" grade (0.20954). Solutions will be evaluated based on the MAP@10 metric. In addition to the solution, please send us the code of the developed recommendation system in GitHub/GitLab. The code must be fully executable and covered with comments. Please specify instructions for running the code in the README.<br/>
In the code, keep all the solutions that were tested during the thesis work. This way, we can evaluate the volume and number of experiments conducted. However, please remember that more experiments are not always better!<br/>
The pass criterion is a solution with a metric higher than the corresponding "Pass" benchmark (MAP@10 > 0.20954) on the Kaggle platform and the submission of reproducible solution code.<br/>
To obtain a "B" grade, in addition to the "Pass" benchmark, you need to complete any two items from the following:
- A hybrid approach to the recommendation system.
- Complete all items from the "Improvements and Additions to the Task" section.
- Structured code (the main logic of the code should be placed in classes/methods/functions + meaningful naming of entities and comments).
- The quality of the model should not change depending on the seasonality.<br/>

To obtain a "A" grade, you need to score > 0.25 + complete all the items for the "B" grade, except for "Improvements and Additions to the Task." A significant bonus in evaluation will be the completion of the "Improvements and Additions" item and the correct assessment of the solution offline on cross-validation.
