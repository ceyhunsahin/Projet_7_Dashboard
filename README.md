# Projet_7_Dashboard
<h3> Description de projet </h3>
Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite développer un modèle de scoring de la probabilité de défaut de paiement du client pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Elle décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

Here is a summary of folder available on it:
It contains all the developpement work for the dashboard (dash plotly code + design.py + assets + csv file + 
json files(for xgboost model) + png files + requirements.txt).

Dashboard is available on site: [Dashboard](https://oc-dashboard-home-risk.herokuapp.com/)

For work on Dashboard, it needs to open Flask-API repostory which was explained in another repo [Projet7_Flask_API](https://github.com/ceyhunsahin/Projet7_Flask_API).

Please note that it was created specific github repository for deployment on Heroku:

Python librairies needed for the Dashboard are the following: 
python = 3.10 pandas=1.4.3 numpy=1.22.4 scikit-learn=1.1.2 plotly=5.10.0 shap=0.41.0 xgboost=1.6.1 pillow=9.2.0
