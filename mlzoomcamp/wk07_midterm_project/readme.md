

# wk07: Mid_term_project

Board games have not only been a source of entertainment for decades, but a source of solace and escape through the pandemic. It has been estimated that the global market for board games will reach [30 billion USD by 2026 through a market growth of 13% from 2021](https://www.arizton.com/market-reports/global-board-games-market-industry-analysis-2024). <br />

To effectively cater to this growing market base, we need to understand what factors drive enjoyment for game enthusiasts. Using a database sourced from [Board Game Geek](https://boardgamegeek.com) and curated by [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-03-12) to include games with a minimum of 50 ratings between the years 1950 and 2016; 10, 532 games were analysed for multiple features and average rating. These features included the number of players, age of players, game category and game mechanism. This project compares both decisiontreeregressor and randomforestregressor models, to predict the average score of a game based upon selected features. This may prove useful for local game board stores or game board cafes to purchase and provide games with the greatest likelihood of success with their patrons, rather than spending funds on games with low interest. 


## To set up the environment

#### Using pipenv, run: 
pipenv install pipenv shell

## Docker Setup

#### To install Docker click [here](https://github.com/jazwilson/workbook/blob/main/mlzoomcamp/wk07_midterm_project/Dockerfile) 
run:    docker build -t boardgame-rating-prediction

#### To run the Docker instance: 

docker run -it -p 9696:9696 boardgame-rating-prediction:latest

## Deployment: AWS Elastic Beanstalk Deploy

#### To deploy eb app run (after pipenv shell)
#### Note: Replace us-west-1 with your closest/cheapest server 
eb init -p docker -r us-west-1 rating-serving

#### To test locally you can run: 

eb local run --port 9696

#### To run a gameboard score prediction in another terminal window: 

pipenv shell python predict-test.py

#### To deploy on aws run

eb create rating-serving-env