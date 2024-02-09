# Team Process Mapping Take-Home Task: Binbin Chen

Goal: In this pre-test, you will first read brief selections from two social science papers (Step 1). You will then go through an end-to-end implementation of a feature and apply it to a dataset of team conversations (Step 2). Finally, you will write a reflection on how well you think this feature extractor performed on the data, as well as how well it performs in operationalizing social science constructs (Step 3).

The idea behind this task is to give you a flavor of the scope of our work — to show how we take inspiration from social science, then apply these ideas in a computational way.

Please write your reflection in this README document.

## 1. High-Level Questions
1a. Which dataset did you choose?

> csopII_conversations_withblanks.csv

1b. What method(s) did you choose? In 1-2 sentences each, describe your sentiment analysis method(s).

> I used the RoBERTa-base model for vanilla result and built k-means clusters according the team positivity metrics, and updated the result based on the clusters.

1c. Does your method capture any of the ideas from Troth et al. and West et al.? If so, which ones?

> Yes, some content of West et al. and my dataset features are correlated like score, efficiency, duration, and difficulty. As West introduced, Teams with higher levels of POB capacities are supposed to have better team optimism, satisfaction, efficacy, resilience, and lower conflicts. So I assumed that there is a mapping between sentiment results and these team performance metrics. 
>
> But I don't see much related emotional skills from the dataset, which can also be helpful if there'll features like how others measure each individual's emotion self-management skills or their emotion self-awareness.

1d. Compared to how Troth et al. and West et al. measured positivity, what are some strengths and weaknesses of your approach?

> strength:
>
> 1. Troth and West measure team positivity in a more general way, while it could describe team positivity in a long time range, the sensitive analysis method can measure this in chat-level and conversation-level, which is more specific for teams to reflect.
> 2. Compared to methods using survey and self-recording as the major source, which would introducing bias and subjectivity, my approach uses concrete numerical and textual data in a new quantitative perspective, and it also supports processing of vast amounts of data rapidly without the need for time-consuming manual assessments. It's not only good for scalability and efficiency, but also maintains objectivity.
> 3. For my method, the data collection ways can be real-time, so we don't need to wait for the human rating and survey collections.
> 4. Given the quantitative scoring results, my approach facilitates the statistical analysis of qualitative data, allowing for the integration of sentiment analysis with other quantitative performance metrics.
>
> weaknesses:
>
> 1. While my approach can identify general sentiment and potentially some emotional states, it may not accurately capture specific psychological constructs like self-efficacy, resilience, and optimism without substantial customization and training on relevant datasets, since it's highly dependent on training data and has assumption that the dataset are from same pattern or distribution.
> 2. It's not sensitive to certain context when it's not trained on organizational or team-based communications. Like the *jury_conversations_with_outcome_var* dataset, if we just used the pre-trained model, it will not measure the uncommon or various mood while given same word "awesome".
> 3. The model is not always right and it always has bias when it's susceptible to biases present in the training data. And we also have to pay attention to overfitting and under-fitting models, which may cause more time to deal with it.
> 4. For deep learning methods, analyzing team conversations raises significant privacy and ethical concerns, especially regarding consent, data handling, and the interpretation of findings. There's also the risk of misinterpretation leading to unintended consequences for individuals within teams.

## 2. Method evaluation
Next, we would like you to consider how you would evaluate your method. How do you know the classification or quantification of emotion is “right?” Try to think critically!

2a. Open up your output CSV and look at the columns you generated. Do the values “make sense” intuitively? Why or why not?

> Yes, that makes sense in a general way. The model can classify when there're some emotional characters for the text, but when some contextual indications behind the words, the model will turn to raise the probability of neutral a lot.

2b. Propose an evaluation mechanism for your method(s). What metric would you use (e.g., F1, AUC, Accuracy, Precision, Recall)?

> 1. Classification metrics
>    1. Yes, I will use F1-score, Accuracy, Precision, Recall for the classification task at the first step, considering the imbalance possibilities of the dataset for the nature of conversation, I will pay more attention to F1-score and treat it as the major metric.
>    2. Then instead of using prediction labels directly, I could just use $R^2$ and $MSE$ for evaluating probabilistic predictions, which provide a measure of goodness of fit and help me to adjust the model more slightly.
> 2. Speed: I should also record the training and predicting time that each model will cost, in case some model requires much more time complexity but don't have remarkable performance improvement.
> 3. Comparison to baseline models: After testing and evaluating each model's performance, I should then compare the result to the original model to see whether we have significant improvements.
> 4. Human feedbacks (my labels): If I have more time to do it, I'm supposed to make labels for a part of the dataset as for the model validation. In this way, I can have my own measurement for this, and adjust hyper-parameters more appropriately.

2c. Describe the steps you would take in evaluating this method. Be as specific as possible.

> 1. First, make labels for a subset of data and treat it as validation set.
> 2. Use probabilistic predictions of each model for hyper-parameter adjustment in validation set given the MSE loss.
> 3. Then use all classification metrics and efficiency metrics to select best model, and compare it to the base model to find significant improvements.
> 4. Measure the correlation and difference between my approach and traditional methods like West and Troth, then try to incorporate them for better measurements. 
> 5. Collect more related paper and proper data for it, iterate to next-round model optimization.

2d. Given the nature of these datasets, what challenges do you anticipate that you may encounter during evaluation? How would you go about resolving them?

> 1. Currently, we have no trustful label for the dataset, which leads me to do unsupervised learning for team performance features. And at first, it's safe to use pre-trained model in this way.
> 2. There's some uncommon conversation context given the various team background, and we have to adjust certain weights for trigger words like "awesome", or we can fine-tuned the tokenizer for different word embedding.

## 3. Overall reflection

3a. How much time did it take you to complete this task? (Please be honest; we are looking for feedback to make sure the task is scoped appropriately, as this is one of the first times we’re using this task.)

> 9 hours for main task, but nearly 6 hours more for dealing with other stuffs (like debugging for the incompatibility between transformers and conda environment)

3b. Finally, provide an overall reflection of your experience. How did you approach this task? What challenge(s) did you encounter? If you had more time, what are additional extensions, improvements, or tests that you would want to implement?

> Generally, this task is interesting and deserves more time for it! Since the task requirement is detailed, so I can approach this step by step. The most annoying challenge is the incompatibility that I mentioned before, while others are common in a machine learning task.
>
> If I have more time, I will first do the label things which help me evaluate the models and validate my thoughts intuitively. Then dealing with another dataset, I'll fine-tuned the models more to match the certain context appropriately. Also do more feature engineering, like PCA for team-feature decomposition. If it's meaningful, we can develop an agent to do the data collection, prediction, evaluation, and optimization iteratively.
