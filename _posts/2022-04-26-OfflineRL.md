---
layout: post
title: Offline RL for Robotics
date: 2022-04-26 11:12:00-0400
description: 
tags: Offline_RL Robotics
comments: true
categories: 
---

<!-- This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine. You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`. If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph. Here is an example:

$$
\sum_{k=1}^\infty |\langle x, e_k \rangle|^2 \leq \|x\|^2
$$

You can also use `\begin{equation}...\end{equation}` instead of `$$` for display mode math.
MathJax will automatically number equations:

\begin{equation}
\label{eq:cauchy-schwarz}
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
\end{equation}

and by adding `\label{...}` inside the equation environment, we can now refer to the equation using `\eqref`.

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) that brought a significant improvement to the loading and rendering speed, which is now [on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php). -->

Offline RL allows the agent to learn state value function and/or policy through previous collected batch data. The agent does not need to interact with the environment, which could be costly and dangerous for real-world scenarios. 

We will be briefly introducing a few Offline RL algorithms. You can also read [this tutorial](https://arxiv.org/pdf/2005.01643.pdf) to get a larger picture of this topic.

&nbsp;
#### Offline RL algorithms introduction

##### 1. [TD3_BC](https://arxiv.org/pdf/2106.06860.pdf)
The fist work we will introduce is [TD3_BC](https://arxiv.org/pdf/2106.06860.pdf). TD3-BC, as its name suggested, has both the component of a well-known actor-critic algorithm [TD3](https://arxiv.org/pdf/1802.09477.pdf) and behavior cloning.

$$
\pi = \underset{\pi}{\operatorname{argmax}} \mathbb{E}_{s,a \sim \mathcal{D}} \left[\lambda Q(s, \pi(s)) - [\pi(s)-a]^2 \right]
$$

comparing with the policy of TD3

$$
\pi = \underset{\pi}{\operatorname{argmax}} \mathbb{E}_{s \sim \mathcal{D}} \left[Q(s, \pi(s)) \right]
$$

the only noticeable differences are:

1. Behavior cloning loss term $$[\pi(s)-a]^2$$

2. normalization term $$\lambda$$

The normalization term $$\lambda$$ is 

$$
  \lambda = \frac{\alpha}{\frac{1}{N} \sum_{s_i, a_i} |Q(s_i, a_i) |} 
$$

In addition, note that in TD3_BC, the actions $a$ are also from dataset $\mathcal{D}$, this is also a difference between online RL TD3. 

In addtion, TD3_BC also normalize the states as

$$
s_i = \frac{s_i-\mu_i}{\sigma_i+\epsilon}
$$

##### 2. [CQL](https://arxiv.org/pdf/2006.04779.pdf)

Conservative Q-Learning (CQL) will be our next paper to be introduced. CQL learns a lower bounded Q values to prevent the overestimation problem.

First let's take a look at the general policy iteration method which contains policy evaluation and policy improvement.

<!-- $$ \hat{Q}_{k+1} = 
\underset{Q}{\operatorname{argmin}} \mathbb{E}_{s,a,s' \sim \mathcal{D}} 
\left[ r(s,a) + \gamma \mathcal{E}
\right]
$$

$$$$ -->

{% include figure.html path="assets/img/screenshots/CQR_eq1.png" class="img-fluid rounded" zoomable=true %}

here $$k$$ denotes iteration variable. If we apply this naive approach to offline RL problem, we will suffer from distribution shit. 
&nbsp;
#### Offline RL for robotics
