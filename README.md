## Reinforcement learning portfolio optimization

### TODO
- [ ] File Structure (like this image))
- [ ] <img width="302" alt="Screenshot 2023-09-15 at 12 14 29 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/cf89d19a-7bcf-4d42-b111-b875183c8056">
- [ ] Backend Fast API
- [ ] FrontEnd with D3

## About

- Reinforcement learning has emerged as a promising approach to solve a variety of complex problems. In this project, we focus on designing a multi-stock portfolio optimization/trading environment using OpenAI's Gym library.
- The aim is to demonstrate how to develop a custom environment and write a policy and neural network architecture for the learning agent. To accomplish this, we collected a dataset containing Open, High, Low, Close, and Volume data points at 5-minute intervals for 30 stocks from the Indian stock market along with the Sensex index values.
- We also added signals for buy and sell based on other Algo-trading strategies such as moving average crossovers, resulting in a total of 50 features for each stock at each time step. Our observation space has a size of (number_of_stocks, number_of_timesteps, number_of_features).
- We then spliced the data into the right window based on the frame_bound variable, scaled the data, and created a custom environment for our trading task. The step method in our environment is the most important, as it updates the current tick and takes the action for the next time step.
- This project contributes to the field of reinforcement learning by providing insights into designing custom environments and policies for multi-stock portfolio optimization/trading, and showcases how to utilize OpenAI's Gym library and StableBaselines3 library for training RL models.
- This project was inspired from [MultiStockRLTrading](https://github.com/Akhilesh-Gogikar/MultiStockRLTrading)

## Proposed System

- The proposed system is a custom environment for multi-stock portfolio optimization and trading using reinforcement learning. The system uses OpenAI's Gym environment to create a simulation environment for algorithmic trading.
- The reinforcement learning algorithm updates the policy of the agent over time to achieve higher rewards, with the objective of maximizing portfolio returns while minimizing risk. The system includes a custom policy and neural network architecture to optimize the learning process. The dataset used for training and testing consists of 5-minute intervals for 3 stocks along with an index fund for a total of 4 assets.
- The system is designed to demonstrate the process of designing a custom environment, policy, and neural network for reinforcement learning in the context of multi-stock portfolio optimization and trading.
- The stocks are preprocessed by modifying the attributes of the datasets. The timestamp attribute of the stocks is converted into the datetime data type and additional attributes of the dataset are removed such as the name and tokens.
- Additional attributes are generated for the environment which has been defined for the multistock environment. These attributes are used for calculating the various parameters of the stocks to generate actions and rewards for the agent.

### Pre-Requisite

- Window 7/10/11
- Anaconda/Miniconda
- Jupyter Notebook/VScode
- Python v3.9.16
- Node JS v18
- NPM

## Procedure

### Website

<img width="1800" alt="Screenshot 2023-09-14 at 11 07 57 PM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/a4ae0eed-9e48-4d8a-89fb-e9f59f23784c">

1. Download and extract the `jcomp.zip`  to your desired folder (say `dest`)
2. In a new Powershell terminal

```powershell
cd path/to/dest/jcomp/stonks
npm i
npm run dev
```

1. Website would be hosted at [http://localhost:3000/](http://localhost:3000/)
2. (Alternative) We hosted the website at vercel and can be visted at [STONKS (stonks-five.vercel.app)](https://stonks-five.vercel.app/)

---

### RL Agent

1. Download and extract the `code.zip`  to your desired folder (say `dest`)
2. In `code` folder, open `environment.yml`
3. Scroll to last line and change `subhr` to your windows username.

```yaml
prefix: C:\Users\<your username here>\.conda\envs\multi_stock_rl_trading
```

1. In a new Anaconda Powershell

```powershell
cd /path/to/dest
conda env create --file=environment.yml
```

1. Open your choice of Code editor, set your kernal to `multi_stock_rl_trading` 
2. We have provided with sample data.
3. (Alternative) Get data (in `.json` format)for your intrested company from the website and replace the sample sata at `/dest/code/test_data`. 
4. Make sure there are ***exactly 3 companies*** , no more, no less. 
5. Run all the Code blocks of `main.ipynb` 
6. DONE

---
### Experimental Results

- To evaluate the performance of our RL model on the multi-stock portfolio optimization and trading task, we conducted experiments on historical stock market data. We used a dataset consisting of (Open, High, Low, Close, Volume) data points at 5-minute intervals for 3 stocks in the Indian stock market along with an index fund, making for a total of 4 assets.

The results of the tested models are:

-  suppression rate=66%, transaction cost=0

<img width="823" alt="Screenshot 2023-09-15 at 12 14 38 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/5a5e7540-8c5b-4937-b363-5842935d9227">

<img width="817" alt="Screenshot 2023-09-15 at 12 14 45 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/911998ff-643a-4f5d-bf7d-b3a2a3089aac">

<img width="810" alt="Screenshot 2023-09-15 at 12 14 52 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/452cf1e4-a9bf-46c0-ac16-b8306af09d87">

<img width="801" alt="Screenshot 2023-09-15 at 12 14 59 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/00459b12-24c4-4a73-b08e-11d99c67f56c">

-  suppression rate=50%, transaction cost=0.01

<img width="827" alt="Screenshot 2023-09-15 at 12 15 07 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/9a4d4700-977c-4bc4-b502-4422a663b0dc">

<img width="800" alt="Screenshot 2023-09-15 at 12 15 13 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/f8b1f5af-8a49-42a2-9c88-27c37541538c">

<img width="812" alt="Screenshot 2023-09-15 at 12 15 23 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/fd6f7a38-1be0-4358-94c5-cd21d95ba2d4">

<img width="804" alt="Screenshot 2023-09-15 at 12 15 31 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/00140e1f-b598-47ea-889e-0bd3acad4907">

- suppression rate=0.75, transaction cost=0.05

<img width="820" alt="Screenshot 2023-09-15 at 12 15 39 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/5fcbfa0d-dba4-4de0-8285-6d6a74a3b031">

<img width="819" alt="Screenshot 2023-09-15 at 12 15 46 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/3996c491-9871-449d-bb05-a861177362fe">

<img width="814" alt="Screenshot 2023-09-15 at 12 15 51 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/2b43fb82-4746-4550-8d8c-0467a9b7f075">

<img width="765" alt="Screenshot 2023-09-15 at 12 15 58 AM" src="https://github.com/metal0bird/Reinforcement_learning_portfolio_optimization/assets/71923741/d15fd586-bd85-4def-829a-eddf8f3964a9">

- From the observed results, PPO and SAC  has earned the highest reward for suppression rate=66% and transaction cost=0 and A2C has earned the highest reward for suppression rate=50% and transaction cost=0.01.
    
Each model has predicted different actions for the test datasets which obtains different rewards.
The suppression rate=75% and transactional cost=0.05 has generated the least rewards as a large proportion of the amount in the environment is reduced due to the transaction cost.



