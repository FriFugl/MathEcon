# Least Squares Monte Carlo methods for American option pricing
This is an implementation of the least-squares monte carlo method developed by Longstaff & Schwartz to price american style options. It has been implemented to be capable of pricing American style options.

* `_processes` contains a class representing a stochastic process.
* `_short_rate_models` contains classes of short rate models to simulate short rate trajectories, calculate zero coupon bond prices and swap rates.
* `_stock_path_models` contains a class for the Geometric Brownian Motion process.
* `_LSM` contains classes for Least Squares Monte Carlo methods for option pricing.

## Short rate models
### The Vasiček model
This model has been implemented with the following dynamics

$$dr_{t} = (b - ar_{t})dt + \sigma dW_{t} \quad a>0.$$

It has an affine term structure, and ZCB prices can be represented as

$$p(t,T)=e^{A(t,T)-B(t,T)r_{t}},$$
where
$$B(t,T) = \frac{1}{a}\left(1-e^{-a(T-t)}\right),$$
$$A(t,T) = \frac{\left(B(t,T)-T+t\right)\left(ab-\frac{1}{2}\sigma^{2}\right)}{a^{2}} - \frac{\sigma^{2}B^{2}(t,T)}{4a}.$$

#### Simulation
Given parameters $r_{0}$, $a$, $b$, $\sigma$ and a time interval $[0,T]$ discretized by $0=t_{0}<t_{1}<\cdots<t_{n-1}<t_{n}=T$, it can be simulated using an exact scheme

$$r_{t_{i+1}} = r_{t_{i}}e^{-a(t_{i+1}-t_{i})} + \frac{b}{a}\left(1-e^{-a(t_{i+1}-t_{i})}\right) + \sigma\sqrt{\frac{1}{2a}\left(1-e^{-2a(t_{i+1}-t_{i})}\right)}z_{i+1},$$

or an Euler scheme (equivalent to a Milstein scheme in this case)

$$r_{t_{i+1}} = r_{t_{i}} + (b-ar_{t_{i}})(t_{i+1}-t_{i}) + \sigma \sqrt{t_{i+1}-t_{i}}z_{i+1}.$$

where $z_{i}$ is an i.i.d. sequence of standard normal random variables.

### Usage
Here is an example of using the Vasiček model from `_short_rate_models`
```
from _short_rate_models import VasicekModel
VasicekModel = VasicekModel(a=1, b=0.05, sigma=0.04) #Initiate model

# M = number of discretization points, N = number of paths, seed is a keyword argument
simulated_short_rates = VasicekModel.simulate(r_0=0.03, T=10, M=120, N=5, method='exact', seed=10)
```
which will create a $N\times M$ pandas dataframe such that each row corresponds to the trajectories of the short rate. Example result is plotted below.
```
plt.figure(figsize=(6, 4))
for i, row in simulated_short_rates.iterrows():
    plt.plot(simulated_short_rates.columns, row, label=f'Trajectory {i+1}', alpha=0.7)
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$r_{t}$', fontsize=10).set_rotation(0)
plt.title('Simulation of 5 short rates in the Vasiček model, seed = 10', fontsize=10)
plt.show()
```
![alt text](https://github.com/FriFugl/MathEcon/blob/setup/demo_files/vasicek_example.png?raw=true)
