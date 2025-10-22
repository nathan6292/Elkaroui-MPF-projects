# 🧮 Monte Carlo Methods for Basket Option Pricing

This project explores and implements several **Monte Carlo techniques** to price a **European call option on a basket of correlated assets**.  
It combines mathematical rigor, numerical optimization, and modern computational tools to achieve high accuracy and performance.

## 🚀 Overview

The notebook systematically develops and compares different Monte Carlo approaches:

1. **Classical Monte Carlo (NumPy)** – a baseline implementation for reference.  
2. **Adaptive Monte Carlo** – automatically increases simulations until a target confidence interval width is reached.  
3. **GPU-Accelerated Monte Carlo (PyTorch)** – leverages GPU computation for massive parallelization.  
4. **Variance Reduction Techniques**:
   - **Antithetic Variables**
   - **Control Variates** using a geometric basket option as control.

Each method is evaluated in terms of **execution time**, **variance**, and **confidence interval width**.

## ⚙️ Key Features

- Fully vectorized **PyTorch** implementation for efficient GPU computation  
- Dynamic precision control through **adaptive Monte Carlo**  
- Integrated **variance reduction** techniques  
- Automatic performance comparison across all methods  
- Clear, modular code suitable for academic or professional extension  

## 📊 Results Summary

| Method | Variance | Elapsed Time (s) | Notes |
|:--|:--:|:--:|:--|
| NumPy | High | Slow | Baseline method |
| PyTorch | Similar | **Much faster** | GPU acceleration |
| NumPy Antithetic | Lower | Medium | Exploits mirrored paths |
| PyTorch Antithetic | Lower | Fast | Combines GPU + Antithetic |
| Control Variate (PyTorch) | Lower | Fast | Good accuracy/time trade-off |
| Control Variate + Antithetic (PyTorch) | **Lowest** | **Fastest** | Best accuracy/time trade-off |

## 🧠 Mathematical Background

We consider a basket of assets following correlated geometric Brownian motions:

$$
dS_t^i = r S_t^i \, dt + \sigma_i S_t^i \, dW_t^i, \quad \text{with } \text{Corr}(dW_t^i, dW_t^j) = \rho_{ij}
$$

The European call option payoff is:

$$
\max\left(\sum_i w_i S_T^i - K, 0\right)
$$

Monte Carlo simulations estimate the discounted expected payoff, incorporating **antithetic variables** and **control variates** to reduce variance.

## 💻 Technologies Used

- **Python 3.10+**
- **NumPy**
- **PyTorch**
- **Matplotlib**
- **SciPy**

## 📈 Conclusion

This project demonstrates how **combining adaptive algorithms, GPU computation, and variance reduction** can significantly enhance both **speed** and **accuracy** in option pricing.  
It provides a foundation for more advanced applications such as **American**, **Asian**, or **path-dependent options**, and can easily integrate **deep learning–based control variates** in future research.
