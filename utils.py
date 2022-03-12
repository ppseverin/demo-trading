from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from math import sqrt
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

class TradingTools():
    def __init__(self):
        pass
    
    def generate_tickbars(self,data, frequency=1000):
        ticks = data.copy()
        ticks.columns = ["DateTime","Bid","Ask"]
        times = ticks["DateTime"].to_numpy()
        prices = ticks["Bid"].to_numpy()
        ask = ticks["Ask"].to_numpy()
        it = 0
        _d=[]
        _o=[]
        _h=[]
        _l=[]
        _c=[]
    #     _bid=[]
        _ask=[]

        for i in range(frequency, len(prices), frequency):
            _d.append(times[i-1])                       # time
            _o.append(prices[i-frequency])               # open
            _h.append(np.max(prices[i-frequency:i]))     # high
            _l.append(np.min(prices[i-frequency:i]))     # low
            _c.append(prices[i-1])                       # close
    #         _bid.append(prices[i-1])                     # bid
            _ask.append(ask[i-1]) 

            it += 1

        cols=["DateTime","Open","High","Low","Close","Ask"]

        res = pd.DataFrame(columns=cols)
        res.DateTime = _d
        res.Open = _o
        res.High = _h
        res.Low = _l
        res.Close = _c
    #     res.Bid = _bid
        res.Ask = _ask
        return res

    def get_scaler_env(self,env,ops):
        """
        Returns scikit-learn scaler object to scale the states
        """
        states=[]
        for _ in range(env.n_step):
            action1 = np.random.choice(env.action_space)
            action2 = np.random.choice(env.action_space)
            state, reward, done, info = env.step(action1,action2,ops=ops)
            states.append(state)
            if done:
                break

        scaler = StandardScaler()
        scaler.fit(states)
        return scaler
    
    def get_scaler_agent(self,agent):
        """
        Returns scikit-learn scaler object to scale the states
        """
        states=[]
        agent.reset()
        total = agent.n_step-1
        for _ in range(agent.n_step):
            ops = []
            prev_state = agent._get_obs()
            state, reward, done, info = agent.act(prev_state,ops=ops,scaler=True)
            states.append(state)
            print('\r'+ "{:.4f}%".format(np.round(agent.cur_step/(agent.n_step-1)*100,decimals=4)) , end=' ')
            if done:
                break
                
        del ops
        scaler = StandardScaler()
        scaler.fit(states)
        return scaler
    
    def get_scaler_agent_2(self,agent):
        """
        Returns scikit-learn scaler object to scale the states
        """
    #     ops=[]
        states=[]
        agent.reset()
        total = agent.n_step-1
        for _ in range(agent.n_step):
            t0 = datetime.now()
            ops=[]
            prev_state = agent._get_obs()
            state, reward, done, info = agent.act(prev_state,ops=ops,scaler=True)
    #         state = state.flatten()
            states.append(state.flatten())
            total_seconds = (datetime.now()-t0).total_seconds()
            print('\r'+ "{:.4f}% --- {:.2f} it/s".format(np.round(agent.cur_step/(agent.n_step-1)*100,decimals=4),
                                                        1/total_seconds) , end=' ')
            if done:
                break
        del ops
        scaler = StandardScaler()
        scaler.fit(states)
        return scaler

    def create_window(self,data,window_size = 1):
        data.columns = map(str.capitalize,data.columns)
        data_s = data.copy()
        data_s.drop(columns={"Open","High","Low","Close","Ask"},inplace=True)
        data_s = pd.DataFrame(data_s.values,index=data_s.index)
        data_c = data.copy()
    #     data_c.drop(columns={"Volume"},inplace=True)
    #     data_s = data_s[[0,1,2,3,4,5,6]].values
    #     pd.DataFrame().rename()
        for i in range(window_size):
            data_c = pd.concat([data_c, data_s.shift((i + 1))], axis = 1)

        data_c.dropna(axis=0, inplace=True)
        data_c.drop(columns={"Open","High","Low"},inplace=True)
        return(data_c)

    def plots(self,run,ops,ops_name,candles,main_folder,data_folder,img_folder,readme=None,plot=False):
        ut = utils()
        try:
            ut.maybe_make_dir(main_folder)
            ut.maybe_make_dir(f"{main_folder}/"+img_folder)
            ut.maybe_make_dir(f"{main_folder}/"+data_folder)
            del ut
            
            if readme is not None:
                file = open(f"{main_folder}/README.txt","w")
                file.write(readme)
                file.close()
    #         saving_path = f"{main_folder}/"+img_dir
    #         url = r"C:\Users\User\Jupyter python ITS\Desarrollos Python"+data_dir+str(run)+".csv"
    #         data = pd.read_csv(url)
    #         data.drop(columns="Unnamed: 0",inplace=True)
            ops.to_csv(f"{main_folder}/{data_folder}/{ops_name}.csv")
            sell_wr = 100*len(ops[(ops.Pips>=0) & (ops.Type=="sell")])/len(ops[ops.Type=="sell"]) \
                if len(ops[ops.Type=="sell"])>0 else 0.0

            buy_wr = 100*len(ops[(ops.Pips>=0) & (ops.Type=="buy")])/len(ops[ops.Type=="buy"]) \
                if len(ops[ops.Type=="buy"])>0 else 0.0

    #         print("Buy WR: {:.2f}%".format(buy_wr),
    #               "Sell WR: {:.2f}%".format(sell_wr),
    #             "Total pips: {:.2f}".format(data.Pips.sum()),
    #             "%W: {:.2f}%".format(pw(data.Pips)),
    #                   "PF: {:.2f}".format(pf(data.Pips)),
    #                   "E: {:.2f}".format(esp(data.Pips)),
    #                   "MaxDD: {}".format(int(maxdd(data.Pips))),
    #                   "Kratio: {:.4f}".format(kratio(data.Pips)), sep="   ")

            ops["_cumsum"] = ops.Pips.cumsum()
            ops["colors"]=['red' if x <= 0 else 'blue' for x in ops._cumsum]

            cmap = ListedColormap(['r', 'b'])
            norm = BoundaryNorm([-60, 0, 250], cmap.N)

            x = ops.index.tolist()
            y = ops._cumsum.tolist()
            z = np.array(list(y))

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap,norm=norm)
            lc.set_array(z)
            textstr = str("Buy WR: {:.2f}%".format(buy_wr)+
            " Sell WR: {:.2f}%".format(sell_wr)+
            " Total pips: {:.2f}".format(ops.Pips.sum())+
            " %W: {:.2f}%".format(self.pw(ops.Pips))+
            " PF: {:.2f}".format(self.pf(ops.Pips))+
            " E: {:.2f}".format(self.esp(ops.Pips))+
            " MaxDD: {}".format(int(self.maxdd(ops.Pips)))+
            " Kratio: {:.4f}".format(self.kratio(ops.Pips)))

            fig, ax = plt.subplots(nrows=2,ncols=1, clear=True,figsize=(20,15))
    #         ax[0].yticks(fontsize=12)
    #         ax[0].xticks(fontsize=12)
            ax[0].tick_params(axis="x", labelsize=12)
            ax[0].tick_params(axis="y", labelsize=12)
            ax[0].add_collection(lc)
            ax[0].set_xlim(min(x), max(x))
            ax[0].set_ylim(min(y)-10, max(y)+10)
            ax[0].grid()
            ax[0].hlines(0,min(x),max(x))
            ax[0].text(.125, -0.15, textstr, fontsize=14, transform=ax[0].transAxes)
            ax[0].set_title("Episode n° "+str(run),fontsize=14)
    #         saving_path = r"C:\Users\User\Jupyter python ITS\Desarrollos Python"+img_dir
    #         plt.savefig(saving_path+"\RL_Agent_Episode_"+str(run)+".png");


            d_open=ops[["Type","Open_Time","Open_Price"]].copy()
            d_close=ops[["Type","Close_Time","Close_Price"]].copy()

            d_open.Open_Time = pd.to_datetime(d_open.Open_Time, format="%Y.%m.%d %H:%M:%S.%f")
            d_close.Close_Time = pd.to_datetime(d_close.Close_Time, format="%Y.%m.%d %H:%M:%S.%f")

            candle_open=pd.merge(left=candles.reset_index(),\
                                 right=d_open.rename(columns={"Open_Time":"DateTime"}),how="left").dropna()
            candle_close=pd.merge(left=candles.reset_index(),\
                                  right=d_close.rename(columns={"Close_Time":"DateTime"}),how="left").dropna()

            start=None
            end=None
    #         ax[1].figure(figsize=(20,7))
    #         ax[1].yticks(fontsize=12)
    #         ax[1].xticks(fontsize=12)
            ax[1].tick_params(axis="x", labelsize=12)
            ax[1].tick_params(axis="y", labelsize=12)
            ax[1].grid()
            ax[1].scatter(pd.DataFrame(candle_open[candle_open.Type=="sell"].index.values,\
                                     index=candle_open[candle_open.Type=="sell"].index.values).loc[start:end],\
                        candle_open[candle_open.Type=="sell"].Open_Price.loc[start:end],color="r",s=250,marker="v")

            ax[1].scatter(pd.DataFrame(candle_open[candle_open.Type=="buy"].index.values,\
                                     index=candle_open[candle_open.Type=="buy"].index.values).loc[start:end],\
                        candle_open[candle_open.Type=="buy"].Open_Price.loc[start:end],color="g",s=250,marker="^")

            ax[1].scatter(pd.DataFrame(candle_close[candle_close.Type=="sell"].index.values,\
                                    index = candle_close[candle_close.Type=="sell"].index.values).loc[start:end],\
                        candle_close[candle_close.Type=="sell"].Close_Price.loc[start:end],color="r",s=250,marker="^")

            ax[1].scatter(pd.DataFrame(candle_close[candle_close.Type=="buy"].index.values,\
                                    index = candle_close[candle_close.Type=="buy"].index.values).loc[start:end],\
                        candle_close[candle_close.Type=="buy"].Close_Price.loc[start:end],color="g",s=250,marker="v")
    #         ax[1].vlines(400,train_candles.reset_index().Close.min(),train_candles.reset_index().Close.max())
            ax[1].plot(candles.reset_index().Close[start:end]);

    #         saving_path = r"C:\Users\User\Jupyter python ITS\Desarrollos Python"+img_dir
            fig.savefig(f"{main_folder}/{img_folder}/{ops_name}.png");

            if plot==False:
                fig.clear()
                ax[0].clear()
                ax[1].clear()
                plt.close();

        except FileNotFoundError:
            print("El archivo aun no existe")


    def maxdd(self,pips):
        dd, dd_actual=[],0
        for p in pips:
            dd_actual-=p
            if dd_actual <0:
                dd_actual=0
            dd.append(dd_actual)
        return np.amax(dd)

    def esp(self,pips):
        return pips.mean()

    def pf(self,pips):
        return pips[pips>0].sum()/np.abs(pips[pips<0].sum())

    def pw(self,pips):
        return pips[pips>0].size/pips.size*100

    def kratio(self,pips):
        x = pips.reset_index().index.values.reshape(-1,1)
        y = pips.cumsum()
        lm = LinearRegression()
        lm.fit(x,y)
        pendiente = lm.coef_[0]
        xhat = np.mean(x)
        yhat = np.mean(y)
        xdev = np.sum(np.square(x - xhat))
        ydev = np.sum(np.square(y - yhat))
        xydev = np.square(np.sum((x.flatten()-xhat)*(y-yhat)))
    #     print(xhat," ", yhat," ", xdev, " ", ydev, " ", xydev)
        if xdev == 0:
            return np.inf
        error = sqrt(np.around((ydev - xydev/xdev)/ (len(x)-2), decimals=8))/sqrt(xdev)
    #     print(error)
        return pendiente/(error*x.size)


    
class FraccDiff:
    def __init__(self):
        pass
    
    def getWeights_threshold(self,d, thres):
        # Intitialise inital weight
        w, k = [1.], 1
        while True:
            # Generate weight using previous weight
            w_ = -w[-1]/k*(d-k+1)
            # Break if absolute value of weight is below a prefixed threshold
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def fracDiff(self,series, d, thres=0.001):
        # Generate weights using the generator
        w = self.getWeights_threshold(d, thres)
        df = {}
        # Iterate over each column in the dataframe
        for name in series.columns:
            # Fill in the NaN values previous values
            df_ = pd.Series(series[name].values, index=series.index).fillna(
                method='ffill').dropna()
            x = pd.Series(0, index=df_.index)
            for k in range(w.shape[0]):
                # Apply the generated weights on the lags
                x = x+w[k, 0]*df_.shift(-k)
        # df[name]=x.dropna().copy(deep=True)
        df[name] = x.shift(k).copy(deep=True)
        df = pd.concat(df, axis=1)
        return df

    def findMinD(self,series,thres):
        # Iterate over a range of d
        for d in np.linspace(0, 1, 11):
            # Find the fractionally differentiated series
            df_ = self.fracDiff(series, d, thres=thres).dropna()
            # Get the ADF statistic
            res = adfuller(df_.iloc[:, 0].values, maxlag=1,
                           regression='c', autolag=None)
            # Check if stationary 
            if (res[0] <= res[4]['1%']):
                return d
        return 1.0
    
    def generate_frac_diff_dataframe(self,data,threshold=0.001):
        data_c = data.copy()
        data_c.columns = map(str.capitalize,data_c.columns)
        log_candles = pd.DataFrame(np.log(data_c.Close))
        d = self.findMinD(log_candles,threshold)
        print("The value of d for generating binomial weights for Fractional differentiation: "+str(d))

        l_candles = pd.DataFrame(columns=["Open","High","Low","Close","Ask"])
        l_candles.Open = np.log(data_c.Open)
        l_candles.High = np.log(data_c.High)
        l_candles.Low = np.log(data_c.Low)
        l_candles.Close = np.log(data_c.Close)
        l_candles.Ask = np.log(data_c.Ask)

        Open_FD = self.fracDiff(pd.DataFrame(l_candles.Open),d,threshold)
        High_FD = self.fracDiff(pd.DataFrame(l_candles.High),d,threshold)
        Low_FD = self.fracDiff(pd.DataFrame(l_candles.Low),d,threshold)
        Close_FD = self.fracDiff(pd.DataFrame(l_candles.Close),d,threshold)
        Ask_FD = self.fracDiff(pd.DataFrame(l_candles.Ask),d,threshold)

        l_candles.Close = Close_FD
        l_candles.Open = Open_FD
        l_candles.Low = Low_FD
        l_candles.High = High_FD
        l_candles.Ask = Ask_FD
    #     l_candles.dropna(inplace = True)
        return l_candles
    
class utils():
    def __init__(self):
        pass
    
    def maybe_make_dir(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)