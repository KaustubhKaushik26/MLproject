## End to End Data Science project

import dagshub
dagshub.init(repo_owner='kaustubhkaushik26', repo_name='MLproject', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
