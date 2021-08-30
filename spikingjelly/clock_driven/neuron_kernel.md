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

## CUDA Kernel

The cuda kernel of spiking neuron will regard any input tensor with arbitrary shape as ``shape=[numel/neuron_num, neuron_num]``.

It will calculate ``index`` where ``0 <= index < neuron_num``, and do a for loop with ``mem_offset = 0, neuron_num, 2 * neuron_num, …, numel - neuron_num``. We can find that the time-step ``t`` is ``t = index + mem_offset``. The following figure shows the cuda memory layout:

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
## Leaky-Integrate-and-Fire Neuron (LIF Neuron)

For the LIF neuron, the charge function is 
$$
H[t] = V[t - 1] + \frac{1}{\tau}(X[t] - (V[t - 1] - V_{reset}))
$$
Then the gradients are
$$
\begin{align}
\frac{\partial L}{\partial H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\partial S[t]}{\partial H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\partial L}{\partial H[t+1]}(1 - \frac{1}{\tau}))\frac{\partial V[t]}{\partial H[t]}\\
\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]} \frac{1}{\tau}\\
\frac{\partial L}{\partial V[0]} &= \frac{\partial L}{\partial H[1]} (1 - \frac{1}{\tau})
\end{align}
$$

## Parametric Leaky-Integrate-and-Fire Neuron (PLIF Neuron)

For the PLIF neuron, the charge function is 
$$
H[t] = V[t - 1] + \frac{1}{\tau}(X[t] - (V[t - 1] - V_{reset}))
$$
Then the gradients are
$$
\begin{align}
\frac{\partial L}{\partial H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\partial S[t]}{\partial H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\partial L}{\partial H[t+1]}(1 - \frac{1}{\tau}))\frac{\partial V[t]}{\partial H[t]}\\
\frac{\partial L}{\partial X[t]} &= \frac{\partial L}{\partial H[t]} \frac{1}{\tau}\\
\frac{\partial L}{\partial \frac{1}{\tau}} &= \sum_{t} \frac{\partial L}{\partial H[t]} (X[t] - (V[t - 1] - V_{reset}))=\sum_{t} \frac{\partial L}{\partial H[t]}(H[t]-V[t-1])\tau\\
\frac{\partial L}{\partial V[0]} &= \frac{\partial L}{\partial H[1]} (1 - \frac{1}{\tau})
\end{align}
$$

