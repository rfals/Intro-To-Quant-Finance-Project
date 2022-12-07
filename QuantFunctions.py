import pandas as pd
import numpy as np


class QuantStuff:

    '''
    This class contains all the functions needed to price the Exotic Option.
    The main purpose behind it is to make the Juptyer Notebook more readable.
    All parameters besides the number of simulations are stored in a list.
    '''
    def __init__(self, name= None):
        self.name = name

    def PriceSimulations(self, params, numPrice_simulations, seed):
        '''
        Function to run the Monte Carlo simulations. Outputs two matrices, one for the stock price and one for the variance.
        params: list of parameters
        numPrice_simulations: number of simulations to run
        seed: seed for the random number generator 1 = True, 0 = False
        '''
        if seed == 1:
            np.random.seed(1)

        V_matrix = pd.DataFrame(np.zeros((252*2, numPrice_simulations)))
        S_matrix = pd.DataFrame(np.zeros((252*2, numPrice_simulations)))

        for i in range(numPrice_simulations):
            V_matrix.iloc[0,i] = params[3]
            S_matrix.iloc[0,i] = params[0]


        dW1 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numPrice_simulations)))
        dW2 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numPrice_simulations)))

        for i in range(numPrice_simulations):
            for k in range(1, int(params[7]**-1*params[8])):
                dW1.iloc[k,i] = np.sqrt(params[7])*np.random.randn() 
                dW2.iloc[k,i] = np.sqrt(params[7])*np.random.randn()
                
                S_matrix.iloc[k,i] = S_matrix.iloc[k-1,i] + params[6]*S_matrix.iloc[k-1,i]*params[7] + np.sqrt(V_matrix.iloc[k-1, i])*S_matrix.iloc[k-1,i]*(params[5]*dW1.iloc[k,i] + np.sqrt(1-params[5]**2)*dW2.iloc[k,i])

                V_matrix.iloc[k,i] = V_matrix.iloc[k-1,i] + params[1]*(params[2] - V_matrix.iloc[k-1,i])*params[7] + params[4]*np.sqrt(V_matrix.iloc[k-1,i])*dW1.iloc[k,i] 
                
                if S_matrix.iloc[k,i] < 0:
                    S_matrix.iloc[k,i] = 0
                elif V_matrix.iloc[k,i] < 0:
                    V_matrix.iloc[k,i] = 0

        return S_matrix, V_matrix


    def OptionsPrice(self, params, S_matrix):
        '''
        Function calculates the price of the option using the output of the PriceSimulations function.
        params: list of parameters
        S_matrix: output of PriceSimulations function
        '''

        payoff_list = []
        for column in S_matrix.columns:
            avg_price = S_matrix[column].mean()
            avg_payoff = max((avg_price - params[9]), 0)
            disc_payoff = avg_payoff*np.exp(-params[6]**params[8])
            payoff_list.append(disc_payoff)

        price = sum(payoff_list)/len(payoff_list)
        return price

    def DeltaSimulations(self, params, numGreek_simulations, seed):
        '''
        Function to run the Monte Carlo simulations and get the exotic options delta that is the sensitivity of the option price to the stock price.
        params: list of parameters
        numGreek_simulations: number of simulations to run
        seed: seed for the random number generator 1 = True, 0 = False
        '''
        if seed == 1:
            np.random.seed(1)
        #Sensitivity
        h = 1
        # V-h
        V_matrix_low = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))
        S_matrix_low = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))

        for i in range(numGreek_simulations):
            V_matrix_low.iloc[0,i] = params[3]
            S_matrix_low.iloc[0,i] = params[0] - h
            
        barrier = 0.7*params[0]
        barrier_sprt = pd.DataFrame((np.zeros((252*2, numGreek_simulations))))
        payoff = pd.DataFrame(np.zeros((1, numGreek_simulations)))
        
        dW1 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))
        dW2 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))

        for i in range(numGreek_simulations):
            for k in range(1, int(params[7]**-1*params[8])):
                dW1.iloc[k,i] = np.sqrt(params[7])*np.random.randn() 
                dW2.iloc[k,i] = np.sqrt(params[7])*np.random.randn()
                
                S_matrix_low.iloc[k,i] = S_matrix_low.iloc[k-1,i] + params[6]*S_matrix_low.iloc[k-1,i]*params[7] + np.sqrt(V_matrix_low.iloc[k-1, i])*S_matrix_low.iloc[k-1,i]*(params[5]*dW1.iloc[k,i] + np.sqrt(1-params[5]**2)*dW2.iloc[k,i])

                V_matrix_low.iloc[k,i] = V_matrix_low.iloc[k-1,i] + params[1]*(params[2] - V_matrix_low.iloc[k-1,i])*params[7] + params[4]*np.sqrt(V_matrix_low.iloc[k,i])*dW1.iloc[k,i] 
                
                if S_matrix_low.iloc[k,i] < 0:
                    S_matrix_low.iloc[k,i] = 0
                elif V_matrix_low.iloc[k,i] < 0:
                    V_matrix_low.iloc[k,i] = 0
            
        payoff_list = []
        for column in S_matrix_low.columns:
            avg_price = S_matrix_low[column].mean()
            avg_payoff = max((avg_price - params[9]), 0)
            disc_payoff = avg_payoff*np.exp(-params[6]**params[8])
            payoff_list.append(disc_payoff)

        price_low = sum(payoff_list)/len(payoff_list) 

        # V+h
        V_matrix_high = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))
        S_matrix_high = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))
            
        for i in range(numGreek_simulations):
            V_matrix_high.iloc[0,i] = params[3]
            S_matrix_high.iloc[0,i] = params[0] + h
            
        barrier = 0.7*params[0]
        barrier_sprt = pd.DataFrame((np.zeros((252*2, numGreek_simulations))))
        payoff = pd.DataFrame(np.zeros((1, numGreek_simulations)))
            

        dW1 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))
        dW2 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))

        for i in range(numGreek_simulations):
            for k in range(1, int(params[7]**-1*params[8])):
                dW1.iloc[k,i] = np.sqrt(params[7])*np.random.randn() 
                dW2.iloc[k,i] = np.sqrt(params[7])*np.random.randn()
                
                S_matrix_high.iloc[k,i] = S_matrix_high.iloc[k-1,i] + params[6]*S_matrix_high.iloc[k-1,i]*params[7] + np.sqrt(V_matrix_high.iloc[k-1, i])*S_matrix_high.iloc[k-1,i]*(params[5]*dW1.iloc[k,i] + np.sqrt(1-params[5]**2)*dW2.iloc[k,i])

                V_matrix_high.iloc[k,i] = V_matrix_high.iloc[k-1,i] + params[1]*(params[2] - V_matrix_high.iloc[k-1,i])*params[7] + params[4]*np.sqrt(V_matrix_high.iloc[k,i])*dW1.iloc[k,i] 
                
                if S_matrix_high.iloc[k,i] < 0:
                    S_matrix_high.iloc[k,i] = 0
                elif V_matrix_high.iloc[k,i] < 0:
                    V_matrix_high.iloc[k,i] = 0
            
        payoff_list = []
        for column in S_matrix_high.columns:
            avg_price = S_matrix_high[column].mean()
            avg_payoff = max((avg_price - params[9]), 0)
            disc_payoff = avg_payoff*np.exp(-params[6]**params[8])
            payoff_list.append(disc_payoff)
            

        price_high = sum(payoff_list)/len(payoff_list)

        # Delta
        delta = (price_high-price_low)/(2*h)

        return delta

    def VegaSimulations(self,params, numGreek_simulations, seed):
        '''
        Function to run the Monte Carlo simulations and get the exotic options Vega, that is the sensitivity of the option price to the volatility of the underlying asset.
        params: list of parameters
        numGreek_simulations: number of simulations to run
        seed: seed for the random number generator 1 = True, 0 = False 
        '''
        #Sensitivity
        if seed == 1:
            np.random.seed(1)

        h = 0.01

        # V-h
        V_matrix_low = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))
        S_matrix_low = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))

        for i in range(numGreek_simulations):
            V_matrix_low.iloc[0,i] = params[3] - h
            S_matrix_low.iloc[0,i] = params[0]
            
        barrier = 0.7*params[0]
        barrier_sprt = pd.DataFrame((np.zeros((252*2, numGreek_simulations))))
        payoff = pd.DataFrame(np.zeros((1, numGreek_simulations)))
        
        dW1 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))
        dW2 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))

        for i in range(numGreek_simulations):
            for k in range(1, int(params[7]**-1*params[8])):
                dW1.iloc[k,i] = np.sqrt(params[7])*np.random.randn() 
                dW2.iloc[k,i] = np.sqrt(params[7])*np.random.randn()
                
                S_matrix_low.iloc[k,i] = S_matrix_low.iloc[k-1,i] + params[6]*S_matrix_low.iloc[k-1,i]*params[7] + np.sqrt(V_matrix_low.iloc[k-1, i])*S_matrix_low.iloc[k-1,i]*(params[5]*dW1.iloc[k,i] + np.sqrt(1-params[5]**2)*dW2.iloc[k,i])

                V_matrix_low.iloc[k,i] = V_matrix_low.iloc[k-1,i] + params[1]*(params[2] - V_matrix_low.iloc[k-1,i])*params[7] + params[4]*np.sqrt(V_matrix_low.iloc[k,i])*dW1.iloc[k,i] 
                
                if S_matrix_low.iloc[k,i] < 0:
                    S_matrix_low.iloc[k,i] = 0
                elif V_matrix_low.iloc[k,i] < 0:
                    V_matrix_low.iloc[k,i] = 0
            
        payoff_list = []
        for column in S_matrix_low.columns:
            avg_price = S_matrix_low[column].mean()
            avg_payoff = max((avg_price - params[9]), 0)
            disc_payoff = avg_payoff*np.exp(-params[6]**params[8])
            payoff_list.append(disc_payoff)
            

        price_low = sum(payoff_list)/len(payoff_list)

        # V+h
        V_matrix_high = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))
        S_matrix_high = pd.DataFrame(np.zeros((252*2, numGreek_simulations)))
            
        for i in range(numGreek_simulations):
            V_matrix_high.iloc[0,i] = params[3] + h
            S_matrix_high.iloc[0,i] = params[0] 
            
        barrier = 0.7*params[0]
        barrier_sprt = pd.DataFrame((np.zeros((252*2, numGreek_simulations))))
        payoff = pd.DataFrame(np.zeros((1, numGreek_simulations)))
            

        dW1 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))
        dW2 = pd.DataFrame(np.zeros((int(params[7]**-1*params[8]), numGreek_simulations)))

        for i in range(numGreek_simulations):
            for k in range(1, int(params[7]**-1*params[8])):
                dW1.iloc[k,i] = np.sqrt(params[7])*np.random.randn() 
                dW2.iloc[k,i] = np.sqrt(params[7])*np.random.randn()
                
                S_matrix_high.iloc[k,i] = S_matrix_high.iloc[k-1,i] + params[6]*S_matrix_high.iloc[k-1,i]*params[7] + np.sqrt(V_matrix_high.iloc[k-1, i])*S_matrix_high.iloc[k-1,i]*(params[5]*dW1.iloc[k,i] + np.sqrt(1-params[5]**2)*dW2.iloc[k,i])

                V_matrix_high.iloc[k,i] = V_matrix_high.iloc[k-1,i] + params[1]*(params[2] - V_matrix_high.iloc[k-1,i])*params[7] + params[4]*np.sqrt(V_matrix_high.iloc[k,i])*dW1.iloc[k,i] 
                
                if S_matrix_high.iloc[k,i] < 0:
                    S_matrix_high.iloc[k,i] = 0
                elif V_matrix_high.iloc[k,i] < 0:
                    V_matrix_high.iloc[k,i] = 0
            
        payoff_list = []
        for column in S_matrix_high.columns:
            avg_price = S_matrix_high[column].mean()
            avg_payoff = max((avg_price - params[9]), 0)
            disc_payoff = avg_payoff*np.exp(-params[6]**params[8])
            payoff_list.append(disc_payoff)
            

        price_high = sum(payoff_list)/len(payoff_list)
        vega = (price_high-price_low)/(2*h)
        return vega

if __name__ == "__main__":
    pass



