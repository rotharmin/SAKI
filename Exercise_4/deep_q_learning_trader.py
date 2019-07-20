import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger

class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 2
        self.action_size = 9#10
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999#0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9


        # attributs at beggining
        self.cash_origin = 0
        self.price_a_origin = 0
        self.price_b_origin = 0
        self.last_price_a = 0
        self.last_price_b = 0

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action_a_b = None
        self.last_action_b = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def replay_new(self):
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state in batch:
            target_qval = reward + self.gamma * np.amax(self.model.predict(next_state))
            target = self.model.predict(state)
            target[0][np.argmax(action)] = target_qval
            self.model.fit(state, target, epochs=1, verbose=0)

    def vote_to_int(self, value_of_vote):
        if(value_of_vote == "sell"):
            return -1
        elif(value_of_vote == "hold"):
            return 0
        else:
            return 1

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        order_list = []
        company_list = stock_market_data.get_companies()
        
        # Compute the current state

        stock_data_a = stock_market_data[Company.A]
        stock_data_b = stock_market_data[Company.B]
        price_a = stock_market_data.get_most_recent_price(Company.A)
        price_b = stock_market_data.get_most_recent_price(Company.B)
        vote_a = self.expert_a.vote(stock_data_a)
        vote_b = self.expert_b.vote(stock_data_b)

        sum = portfolio.get_stock(Company.B) + portfolio.get_stock(Company.A)
        if(sum == 0):
            ratio_a = 0
            ratio_b = 0
        else:
            ratio_a = portfolio.get_stock(Company.A) / sum
            ratio_b = portfolio.get_stock(Company.B) / sum

        if self.last_action_a_b is None:
            #only do at first run
            self.cash_origin = portfolio.cash
            self.price_a_origin = price_a
            self.price_b_origin = price_b
            self.last_price_a = price_a
            self.last_price_b = price_b
        
        state = np.array([[#portfolio.cash / self.cash_origin,
                 #ratio_a,
                 #ratio_b,
                 #price_a - self.last_price_a,
                 #price_b - self.last_price_b,
                 self.vote_to_int(vote_a.value),
                 self.vote_to_int(vote_b.value)]])

        print(self.vote_to_int(vote_a.value), vote_a.value)

        """
        state = np.array([portfolio.cash,
                 portfolio.get_stock(Company.A),
                 portfolio.get_stock(Company.B),
                 price_a,
                 price_b,
                 ord(vote_a.value[0]),
                 ord(vote_b.value[0])])
        #print(state)
        """

        
        current_portfolio_value = portfolio.get_value(stock_market_data)
        # Store state as experience (memory) and train the neural network only if trade() was called before at least once
        if self.last_action_a_b is not None and self.train_while_trading:
            
            current_portfolio_value = portfolio.get_value(stock_market_data)

            if(self.last_portfolio_value < current_portfolio_value):
                reward = ((current_portfolio_value / self.last_portfolio_value) * 100)# **2
            elif(self.last_portfolio_value < current_portfolio_value):
                reward = 0
            else:
                reward = -100

            self.memory.append((self.last_state, self.last_action_a_b, reward, state))

            if len(self.memory) > self.min_size_of_memory_before_training:
                self.replay_new()
                #print("finished training minibatch")
           
           #predict state based on old state
            #new_q_val = reward + self.gamma * np.amax(self.model.predict(np.array([self.last_state]))[0])
            #target_f = self.model.predict(state.reshape((1, self.state_size)))
            #target_f[0][np.argmax(self.last_action_a_b)] = new_q_val
            #self.model.fit(state.reshape((1, self.state_size)), target_f, epochs=1, verbose=0)

            #print(predicted_action)
        
        # Create actions for current state and decrease epsilon for fewer random actions
        if ((np.random.rand() <= self.epsilon) and (self.epsilon > self.epsilon_min) and self.train_while_trading):
            action = np.random.randint(self.action_size)
            print(self.epsilon, "random", action)
        else:
            # predict action based on the old state
            #prediction = self.model.predict(state.reshape((1,self.state_size)))
            prediction = self.model.predict(state)
            action = np.argmax(prediction[0])
            if(self.train_while_trading):
                print(self.epsilon, "predicted")
                print(prediction, action)
        
        if(self.epsilon > self.epsilon_min*self.epsilon_decay):
            self.epsilon = self.epsilon * self.epsilon_decay
        #print(self.epsilon)
        # Save created state, actions and portfolio value for the next call of trade()
        self.last_state = state
        self.last_action_a_b = action
        self.last_portfolio_value = current_portfolio_value
        self.last_price_a = stock_market_data.get_most_recent_price(Company.A)
        self.last_price_b = stock_market_data.get_most_recent_price(Company.B)

        #convert action to orderlist
        amount_to_sell_A = portfolio.get_stock(Company.A)
        amount_to_sell_B = portfolio.get_stock(Company.B)
        stock_price_A = stock_market_data[Company.A].get_last()[-1]
        amount_to_buy_A = int(portfolio.cash // stock_price_A)
        stock_price_B = stock_market_data[Company.B].get_last()[-1]
        amount_to_buy_B = int(portfolio.cash // stock_price_B)

        if(action == 0):
            invest = int(portfolio.cash // 2)           
            amount_to_buy_A = int(invest // stock_price_A)
            order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy_A))          
            amount_to_buy_B = int(invest // stock_price_B)
            order_list.append(Order(OrderType.BUY, Company.B, amount_to_buy_B))
        elif(action ==1):
            order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy_A))  
            order_list.append(Order(OrderType.SELL, Company.B, amount_to_sell_B))
        elif(action ==2):
            order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy_A))  
            # hold B
        elif(action == 3):
            order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell_A))  
            order_list.append(Order(OrderType.BUY, Company.B, amount_to_buy_B))
        elif(action ==4):
            order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell_A))   
            order_list.append(Order(OrderType.SELL, Company.B, amount_to_sell_B))
        elif(action == 5):
            order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell_A)) 
        elif(action == 6):
            order_list.append(Order(OrderType.BUY, Company.B, amount_to_buy_B))
        elif(action == 7):
            order_list.append(Order(OrderType.SELL, Company.B, amount_to_sell_B))
        elif(action == 8):
            order_list = order_list
        return order_list

# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    #training_data = training_data.deepcopy_first_n_items(50)
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()
