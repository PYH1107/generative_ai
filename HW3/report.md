This essay will discuss the difference between Machine Learning and GAN with the aspect of cross entropy and KL divergence.

# Definition

## Entropy

Entropy 測量的是概率分佈中的不確定性或信息量

### Shanon Entropy
$$H(p) = \mathbb{E}_p[-\log p] = -\sum_{x \in X} p(x)\log p(x)$$
- 這描述了一個機率分佈的**不確定性**。


### Cross Entropy
$$H(p,q) = \mathbb{E}_p[-\log q] = -\sum_{x \in X} p(x)\log q(x)$$
- 當 $q$是我們的模型預測，而 $p$ 是真實分佈時，cross-entropy 衡量的是模型與真實分佈的差距。
### comparison of  Shanon Entropy and Cross Entropy

| 特性           | Shannon熵                                     | 交叉熵                                                  |
| ------------ | -------------------------------------------- | ---------------------------------------------------- |
| **定義**       | 單一機率分佈的不確定性測量                                | 兩個機率分佈之間的差異測量                                        |
| **數學公式**     | $H(p) = -\sum_{x} p(x) \log p(x)$            | $H(p,q) = -\sum_{x} p(x) \log q(x)$                  |
| **輸入參數**     | 單一分佈 p                                       | 兩個分佈 p 和 q                                           |
| **最小值情況**    | 當分佈確定性為1時(無不確定性)                             | 當 $q = p$ 時，此時等於 Shannon's entropy                   |
| **對稱性**      | 不適用(只有一個分佈)                                  | 非對稱：$H(p,q) \neq H(q,p)$                             |
| **數值範圍**     | 非負值，$[0, log(n)]$                            | 非負值，$[H(p), +∞)$                                     |
| **主要應用**     | 1. 信息理論中測量不確定性<br>2. 決策樹中特徵選擇<br>3. 評估數據壓縮效率 | 1. 分類問題的損失函數<br>2. 測量預測分佈與真實分佈的差異<br>3. 神經網絡訓練中的目標函數 |
| **在GAN中的角色** | 作為理論基礎                                       | 用於判別器的損失函數                                           |
| **與KL散度關係**  | KL散度的一個組成部分                                  | $H(p,q) = H(p) + D_{KL}(p\|q)$                       |
| **實例**       | 公平硬幣拋擲熵 = 1<br>確定性事件熵 = 0                    | 分類器預測值與真實標籤越接近，交叉熵越小                                 |


## Kullback-Leibler Divergence

By definition, we have
$$D_{KL}(p||q) = H(p, q) - H(p)$$

Moreover, we can say$$D_{KL}(p||q) = \mathbb{E}_p[\log\frac{p}{q}] = \sum_{x \in X} p(x)\log\frac{p(x)}{q(x)}$$
- KL Divergence describes 兩個機率分佈之間的差異

## NOTE
1. 當 $q = p$ 時，cross-entropy = Shannon entropy；
2. 當 $q ≠ p$ 時，cross-entropy $\geq$ Shannon entropy
	- 差值正好是KL divergence
	- 這就是為什麼 **minimum of cross- entropy** = **minimum of KL divergence** (當真實分佈 $p$ is fixed)。

# Machine Learning

## Likelihood

考慮一個有 $N$ 筆數據的資料集 $D$：

- 其中 $x_1$ 的標籤是 $+1$（"○"）
- 其他樣本 $x_2, x_3, \ldots, x_N$ 都是 $-1$（"×"）

這個資料集出現的機率可以表示為：

$$P(D|\theta) = \prod_{n=1}^N P(y_n|x_n; \theta)$$

IOW,
- 第 1 筆數據的標籤是 $+1$，機率是 $f(x_1)$
- 其他數據的標籤是 $-1$，機率是 $1 - f(x_n)$
- 這些機率相乘，得到整個資料集 $D$ 出現的可能性（likelihood）

What's more, 對於二元分類問題，我們可以進一步展開：

$$P(D|\theta) = \prod_{n=1}^N P(y_n|x_n; \theta) = \prod_{n=1}^N P(y=+1|x_n; \theta)^{\frac{y_n+1}{2}} \cdot P(y=-1|x_n; \theta)^{\frac{1-y_n}{2}}$$

其中，指數 $\frac{y_n+1}{2}$ 和 $\frac{1-y_n}{2}$ 是為了在 $y_n = +1$ 時只取 $P(y=+1|x_n; \theta)$，在 $y_n = -1$ 時只取 $P(y=-1|x_n; \theta)$。

By defining  $f(x; \theta) = P(y=+1|x; \theta)$，then $P(y=-1|x; \theta) = 1 - f(x; \theta)$。
這樣，likelihood 函數可以重寫為：$$P(D|\theta) = \prod_{n=1}^N f(x_n; \theta)^{\frac{y_n+1}{2}} \cdot (1 - f(x_n; \theta))^{\frac{1-y_n}{2}}$$

### Maximum Likelihood Estimation, MLE

要讓模型學習到最佳的參數，我們希望**最大化這個 likelihood**
That is, 我們的目標是： $$\theta^* = \arg\max_\theta \log P(D|\theta) = \arg\max_\theta \sum_{n=1}^N \log P(y_n|x_n; \theta)$$
- **$\theta$**：模型的參數，例如線性回歸的係數、神經網路的權重等。
- **$\arg\max_\theta$**：我們希望找到最適合的模型參數 $\theta$，使得整個數據集的機率最大。

It is equivalent to optimize:$$\theta^* = \arg\max_\theta \log P(D|\theta) = \arg\max_\theta \sum_{n=1}^N \log P(y_n|x_n; \theta)$$

or：$$\theta^* = \arg\min_\theta -\sum_{n=1}^N \log P(y_n|x_n; \theta)$$,which is equivalent to **Cross-Entropy Error** (namely, cross entropy loss function)

- ref: [Cross Entropy, KL Divergence, and Maximum Likelihood Estimation](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/?utm_source=chatgpt.com)

#### Conclusion
While $p$ is one-hot: $$p(y|x_i) = \begin{cases} 1 & \text{if } y=y_i \\\\ 0 & \text{otherwise} \end{cases}$$

we can simplified cross entropy as:$$H(p,q_\theta) = -\sum_{y \in Y} p(y|x_i) \log q_\theta(y|x_i) = -\log q_\theta(y_i|x_i)$$

Thus, minimizing cross-entropy is equivalent to maximizing likelihood.


## Cross Entropy in Machine Learning

- 在分類問題（如 Logistic Regression、Softmax classifier）中，cross entropy error 衡量的是**真實標籤分佈和模型預測分佈之間的差異**
- cross entropy error in binary classification problems： $$L = -\frac{1}{N} \sum_{i=1}^{N} \big( y_i \log q(y_i | x_i) + (1 - y_i) \log (1 - q(y_i | x_i)) \big)$$
	- 目標是 minimize cross entropy，使模型能夠正確分類樣本

####  Example: Logistic Regression

By setting sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, we have :$$f(x; w) = P(y=+1|x; w) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$其中 $w$ 是模型參數（權重向量）

In contrast, we also obtain$$P(y=-1|x; w) = 1 - f(x; w) = 1 - \sigma(w^T x) = \frac{e^{-w^T x}}{1 + e^{-w^T x}}$$
That is, 
$$P(y|x; w) = \frac{1}{1 + e^{-y \cdot w^T x}}$$

- The likelihood of the dataset $D$ in logistic regression is: $$P(D|w) = \prod_{n=1}^N P(y_n|x_n; w) = \prod_{n=1}^N \frac{1}{1 + e^{-y_n w^T x_n}}$$
- Taking the negative logarithm gives the cross-entropy error:$$\mathbb{E}_{in}(w) = -\frac{1}{N}\sum_{n=1}^N \log\left(\frac{1}{1 + e^{-y_n w^T x_n}}\right) = \frac{1}{N}\sum_{n=1}^N \log(1 + e^{-y_n w^T x_n})$$
#### Cross Entropy Loss in Binary Classification
In binary classification problems, suppose we have $y \in \{0, 1\}$（而不是 ${-1, +1}$），則 cross entropy loss 可以寫為：

$$L = -\frac{1}{N} \sum_{i=1}^{N} \big( y_i \log f(x_i; \theta) + (1 - y_i) \log (1 - f(x_i; \theta)) \big)$$

其中 $f(x_i; \theta)$ 是模型預測 $x_i$ 屬於正類（$y=1$）的機率。

- ref: [機器學習基石](https://www.youtube.com/playlist?list=PLXVfgk9fNX2I7tB6oIINGBmW50rrmFTqf)
# GAN
## Basic Explanation

- GAN由生成器（Generator, G）和判別器（Discriminator, D）組成
- 生成器 $G(z)$：生成假樣本（類似真實樣本）
- 判別器 $D(x)$：判斷輸入$x$是否為真樣本（1 = 真，0 = 假）

## 判別器（Discriminator）損失

> 判別器$D$的目標是區分真實數據$x$和生成數據$G(z)$

- 損失函數： $$L_D = -\mathbb{E}_{x \sim p_{data}} [\log D(x)] - \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z) )\;)]$$
- 第一項 ($-\mathbb{E}_{x \sim p_{data}} [\log D(x)]$)：
	- 當輸入是真實數據 $x$ 時，$D(x)$應該靠近1
	- 通過最小化 $-\log D(x)$ 來實現
- 第二項 ( $-\mathbb{E}_{z \sim p_z} [\log (1 - D(G(z) )\;)]$)：
	- 當輸入是假數據$G(z)$時，$D(G(z))$應該靠近$0$
	- 通過最小化 $\log (1 - D(G(z)))$ 來實現

## 生成器（Generator）損失

> 生成器$G$想要「騙過」判別器$D$，使得$D(G(z))$越接近$1$越好

- 損失函數：$$L_G = -\mathbb{E}_{z \sim p_z} [\log D(G(z))]$$
- 希望$D(G(z))$越接近1，表示判別器被騙了，認為$G(z)$生成的樣本是真實的

## GAN的目標函數

GAN的訓練目標可表示為一個 minimax game：$$\min_G \max_D V(D, G) = \mathbb{E}_{x\sim p_{data}}[\log(D(x))] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]$$
# Comparison of Cross Entropy


| **比較項目**        | **機器學習（分類問題）**            | **GAN（生成對抗學習）**                     |
| --------------- | ------------------------- | ----------------------------------- |
| **目標**          | 訓練分類器，使預測$q(y\|x)$越接近真實標籤 | 生成器：生成難以區分於真實數據的樣本<br>判別器：區分真實與生成樣本 |
| **輸入**          | 已標記的數據$(x,y)$             | 生成樣本$G(z)$和真實樣本$x$                  |
| **損失函數**        | $L = -∑p(y)log q(y)$      | $L_D = -log D(x) - log(1-D(G(z)))$  |
| **optimization* | 最小化損失，提高分類準確率             | 生成器最大化$D(G(z))$，判別器最小化$D(G(z))$     |
- **為什麼GAN也使用 cross entropy**

1. **判別器本質上是個分類器**
    - 它的目標是區分「真實」與「假樣本」，這與監督學習中的二元分類問題類似
2. **生成器的目標是「最大化被判別器認為是真實的機率」**
    - 這與監督學習不同，它是在對抗性訓練中尋找最佳策略
3. 主要差別：
	1. 監督學習中，cross entropy 直接用於標籤對應的真實分佈和預測分佈之間的差異
	2. GAN中，cross entropy 用於對抗性學習，判別器希望最小化，而生成器希望最大化判別器的錯誤率