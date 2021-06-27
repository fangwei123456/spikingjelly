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
We define the FPTT function as
$$
S[1,2,...,T], V[1,2,...,T] = F_{fp}(X[1,2,...,T], V[0])
$$
Thus, we need to define the backward as
$$
\frac{\partial L}{\partial X[1,2,...,T]},\frac{\partial L}{\partial V[0]} = F_{bp}(\frac{\partial L}{\partial S[1,2,...,T]},\frac{\partial L}{\partial V[1,2,...,T]})
$$
According to the forward function, we can get
$$
\begin{align}
	\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]} \frac{\partial H[t]}{\partial X[t]}\\
	\frac{\partial L}{\partial H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\partial S[t]}{\partial H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\partial L}{\partial H[t+1]}\frac{\partial H[t+1]}{\partial V[t]})\frac{\partial V[t]}{\partial H[t]}\\
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
\frac{\partial L}{\partial H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\partial S[t]}{\partial H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\partial L}{\partial H[t+1]}\frac{\partial H[t+1]}{\partial V[t]})\frac{\partial V[t]}{\partial H[t]}\\
\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]}\frac{\partial H[t]}{\partial X[t]}\\
\frac{\partial L}{\partial V[0]} &= \frac{\partial L}{\partial H[1]}
\end{align}
$$

S, V with arbitrary shape will be regraded shape=[numel/neuron_num, neuron_num]

0<= index < neuron_num

mem_offset = 0, neuron_num, 2 * neuron_num, …, numel - neuron_num



t = index + mem_offset



|                      | 0          | 1    |       ...        | neuron_num-1 |
| -------------------- | ---------- | ---- | :--------------: | ------------ |
| 0                    |            |      |                  |              |
| 1                    |            |      |                  |              |
| ...                  | mem_offset |      | index+mem_offset |              |
| numel/neuron_num - 1 |            |      |                  |              |



## Integrate-and-Fire Neuron (IF Neuron)

For the IF neuron, the charge function is 
$$
H[t] = V[t - 1] + X[t]
$$
Then the gradients are
$$
\begin{align}
\frac{\partial L}{\partial H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\partial S[t]}{\partial H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\partial L}{\partial H[t+1]})\frac{\partial V[t]}{\partial H[t]}\\
\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]}\\
\frac{\partial L}{\partial V[0]} &= \frac{\partial L}{\partial H[1]}
\end{align}
$$
The IF neuron needs `H[1,...,T], S[1,...,T]` to calculate gradients.

