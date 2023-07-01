# Model Card

## Model Details
This model is a Random Forest Classifier trained using Grid Search CV by Wesley Giles on June 30th, 2023. Both the Model class and Grid Serach algorithm were provided by the Scikit-Learn python library, version 1.2.2(Buitinck et al. 2013). This model is licensed under the [BSD 3-Clause License](https://opensource.org/license/bsd-3-clause/).

For any additional questions please reach out to Wesley Giles @ [Wesley.Giles@gmail.com](mailto:wesley.giles@gmail.com).

### Citations
- Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., ... & Varoquaux, G. (2013). API design for machine learning software: experiences from the scikit-learn project. arXiv preprint arXiv:1309.0238.

## Intended Use
This model was intended to be used for practicing Deploying a Machine Learning Model using FastApi. While it can be used by students as a study on how this process is achieved, it is not intended to perform any real-world tasks, and all results should be taken as inaccurate, due to the lack of data and precision.

## Training Data
This model was trained on 80% of the census data provided for the project, which is located [here](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv). This data has 35621 rows in total, each with 15 columns. 14 of these columns were transformed and used to predict the `salary` column, which is a binary column indicationg whether or not an individual made more than $50,000 a year in salary.

## Evaluation Data
The evaluation data was the remaining 20% of the aforemention census data held out from training, and should be considered to be a good representation of the whole population.

## Metrics
The three metrics used to evaluate this model's performance were precision, recall, and an f<sub>1</sub> score. The results of the model on the validation data are as follows:

- Precision: 0.6444444444444445
- recall: 0.313015873015873
- f<sub>1</sub>: 0.42136752136752137

It should be noted that the model performed with a significantly improved score in each of these metrics, leading to the belief that significant overfitting was occuring. More research into solutions to this problem will be investigated in the future. The training metrics are as follows:

- Precision: 0.7933631142310147
- recall: 0.992020427705075
- f<sub>1</sub>: 0.8816396000283668


## Ethical Considerations
Given the significant overfitting and disparity in the size of data samples between certain protected characteristics, such as sex, it is not recommended to utilize this model in any real-world application.

## Caveats and Recommendations
More data should be collected to prevent the afformentioned overfitting, as well as improve the overall quality of the model. In additon, as mentioned previously more work should be done to research solutions to the overfitting issues.