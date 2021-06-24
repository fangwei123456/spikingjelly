# neuron_kernel

This md file shows how we define cuda kernel for neuronal forward and backward.

## Definition

We use three equations to describe spiking neurons:
$$
\begin{align}
	H[t] &= f(V[t - 1], X[t])\\
	S[t] &= \Theta(H[t] - V_{th})\\
	V[t] &= \begin{cases}
	H[t]\left( 1 - S[t] \right) + V_{reset}S[t], &~Hard~Reset\\
	H[t] - V_{th}S[t], &~Soft~Reset\\
\end{cases}
\end{align}
$$
We define the forward function as
$$
S[t], V[t] = F_{fp}(X[t], V[t-1])
$$
Thus, we need to define the backward as
$$
\frac{\partial L}{\partial X[t]},\frac{\partial L}{\partial V[t-1]} = F_{bp}(\frac{\partial L}{\partial S[t]},\frac{\partial L}{\partial V[t]})
$$
According to the forward function, we can get
$$
\begin{align}
	\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]} \frac{\partial H[t]}{\partial X[t]}\\
	\frac{\partial L}{\partial V[t-1]} &= \frac{\partial L}{\partial H[t]} \frac{\partial H[t]}{\partial V[t-1]}\\
	\frac{\partial L}{\partial H[t]} &= \frac{\partial L}{\partial S[t]} \frac{\partial S[t]}{\partial H[t]} + \frac{\partial L}{\partial V[t]} \frac{\partial V[t]}{\partial H[t]}\\
	\frac{\partial S[t]}{\partial H[t]} &= \Theta'(H[t] - V_{th})\\
	\frac{\partial V[t]}{\partial H[t]} &= 
	\begin{cases}
		1 - S[t] + (-H[t] + V_{reset})\frac{\partial S[t]}{\partial H[t]}(1-D_{reset}), &~Hard~Reset\\
		1 - V_{th}\frac{\partial S[t]}{\partial H[t]}(1-D_{reset}), &~Soft~Reset\\
	\end{cases}
\end{align}
$$
where
$$
D_{reset} = \begin{cases}
	1, &~Detach~Reset\\
	0, &~Not~Detach~Reset\\
\end{cases}
$$
Finally, we will calculate gradients as
$$
\begin{align}
\frac{\partial L}{\partial H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\partial S[t]}{\partial H[t]} + \frac{\partial L}{\partial V[t]}\frac{\partial V[t]}{\partial H[t]}\\
\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]}\frac{\partial H[t]}{\partial X[t]}\\
\frac{\partial L}{\partial V[t-1]} &= \frac{\partial L}{\partial H[t]}\frac{\partial H[t]}{\partial V[t-1]}
\end{align}
$$

## Integrate-and-Fire Neuron (IF Neuron)

For the IF neuron, the charge function is 
$$
H[t] = V[t - 1] + X[t]
$$
Then the gradients are
$$
\begin{align}
\frac{\partial L}{\partial H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\partial S[t]}{\partial H[t]} + \frac{\partial L}{\partial V[t]}\frac{\partial V[t]}{\partial H[t]}\\
\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]}\\
\frac{\partial L}{\partial V[t-1]} &= \frac{\partial L}{\partial H[t]}
\end{align}
$$
The IF neuron needs `H[t], S[t]` to calculate.