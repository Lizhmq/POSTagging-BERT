## 中文词性标注 - Chinese-RoBERTa-wwm-ext


### 任务介绍

词性标注是指在给定句子中判定每个词的语法范畴，确定其词性并加以标注的过程，这也是自然语言处理中一项非常重要的基础性工作。该项目为中文词性标注：给定中文分词结果，需要预测每个分词的词性标签。例如，训练集中的一条数据为：
```
海内外/Nl  關注/Vt  的/Us  一九九七/Mo  年/Qc  七月/Nt  一/Mo  日/Qc  終於/Dc  來到/Vt  。/Sy
```
那么，任务的输入为：
```
海内外  關注  的  一九九七  年  七月  一  日  終於  來到  。
```
需要预测的输出则为：
```
Nl  Vt  Us  Mo  Qc  Nt  Mo  Qc  Dc  Vt  Sy
```
该项目分两部分，分别为简体中文的词性标注和繁体中文的词性标注。我们对两个任务各训练了一个配置相同的模型。


### 实验设置

我们使用Chinese-RoBERTa-wwm-ext作为基础模型，用其对句子中的每一个字产生上下文相关的嵌入向量表示，然后再用线性层对该嵌入向量进行分类（注：Chinese-RoBERTa对每个字产生向量表示，我们预测一个词的词性时，直接使用第一个字的向量表示。）

Chinese-RoBERTa-wwm-ext的模型大小与RoBERTa-base相同。它由12层隐层维度为768的Transformer层构成，额外添加的线性分类层参数大小为$768 \times |\mathcal{Y}|$，其中$\mathcal{Y}$为标签空间。整个模型的参数大小约为97.6M。

我们将两个数据集都按照8:1:1划分为训练集、验证集和测试集，在两个训练集上各训练一个模型，最大轮数为20。训练时batch size设置为32。

### 实验结果

数据集一（繁体中文）：准确率、F1和各类的指标如下：

|      | Accuracy  | Macro F1  | Average F1  |
| ---- | --------- | --------- | ----------- |
|      | 0.95      | 0.85      | 0.95        |

|      | precision | recall | f1-score | support |
| ---- | --------- | ------ | -------- | ------- |
| 1    | 0.9       | 0.88   | 0.89     | 378     |
| 2    | 0.7       | 0.81   | 0.75     | 63      |
| 3    | 0.99      | 0.99   | 0.99     | 3161    |
| 4    | 0.95      | 0.95   | 0.95     | 7201    |
| 5    | 0.99      | 0.99   | 0.99     | 7130    |
| 6    | 0.97      | 0.96   | 0.96     | 427     |
| 7    | 0.97      | 0.98   | 0.98     | 4097    |
| 8    | 0.81      | 0.8    | 0.8      | 1152    |
| 9    | 0.99      | 0.99   | 0.99     | 6282    |
| 10   | 0.7       | 0.68   | 0.69     | 186     |
| 11   | 0.56      | 0.41   | 0.47     | 161     |
| 12   | 0.63      | 0.58   | 0.6      | 508     |
| 13   | 1         | 0.99   | 0.99     | 8894    |
| 14   | 0.8       | 0.88   | 0.84     | 219     |
| 15   | 0.97      | 0.97   | 0.97     | 47439   |
| 16   | 0.84      | 0.86   | 0.85     | 1635    |
| 17   | 0.85      | 0.86   | 0.86     | 1121    |
| 18   | 0.94      | 0.94   | 0.94     | 28608   |
| 19   | 0.94      | 0.95   | 0.94     | 3390    |
| 20   | 0.62      | 0.54   | 0.58     | 138     |
| 21   | 0.84      | 0.83   | 0.84     | 7479    |
| 22   | 0.96      | 0.96   | 0.96     | 9479    |
| 23   | 0.89      | 0.85   | 0.87     | 1103    |
| 24   | 1         | 1      | 1        | 30021   |
| 25   | 0.94      | 0.93   | 0.94     | 548     |
| 26   | 1         | 0.67   | 0.8      | 6       |
| 27   | 0.97      | 0.98   | 0.97     | 3207    |
| 28   | 0.84      | 0.85   | 0.84     | 512     |
| 29   | 0.98      | 0.98   | 0.98     | 6224    |
| 30   | 0.98      | 0.99   | 0.98     | 6898    |
| 31   | 0.92      | 0.96   | 0.94     | 1415    |
| 32   | 0.94      | 0.93   | 0.93     | 14302   |
| 33   | 0.89      | 0.89   | 0.89     | 5705    |
| 34   | 0.72      | 0.76   | 0.74     | 196     |
| 35   | 0.95      | 0.93   | 0.94     | 226     |
| 36   | 0.96      | 0.95   | 0.96     | 2683    |
| 37   | 0.85      | 0.86   | 0.86     | 2061    |
| 38   | 0.5       | 0.36   | 0.42     | 25      |
| 39   | 0.98      | 0.98   | 0.98     | 939     |
| 40   | 0.61      | 0.47   | 0.53     | 36      |
| 41   | 0.81      | 0.81   | 0.81     | 3259    |
| 42   | 1         | 1      | 1        | 5       |
| 43   | 0.45      | 0.39   | 0.42     | 83      |

数据集二（简体中文）：准确率、F1和各类的指标如下：

|      | Accuracy  | Macro F1  | Average F1  |
| ---- | --------- | --------- | ----------- |
|      | 0.97      | 0.89      | 0.97        |


|      | precision | recall | f1-score | support |
| ---- | --------- | ------ | -------- | ------- |
| 0    | 1         | 1      | 1        | 6       |
| 1    | 0.91      | 0.94   | 0.93     | 1733    |
| 2    | 0.97      | 0.97   | 0.97     | 5091    |
| 3    | 1         | 1      | 1        | 11020   |
| 4    | 1         | 0.99   | 1        | 120     |
| 5    | 0.95      | 0.97   | 0.96     | 696     |
| 6    | 0.94      | 0.91   | 0.92     | 80      |
| 7    | 0.98      | 0.99   | 0.99     | 3471    |
| 8    | 1         | 1      | 1        | 4457    |
| 9    | 0.99      | 1      | 1        | 338     |
| 10   | 0.6       | 0.75   | 0.67     | 93      |
| 11   | 0.96      | 0.96   | 0.96     | 80      |
| 12   | 0.99      | 0.99   | 0.99     | 3146    |
| 13   | 0.85      | 0.73   | 0.79     | 15      |
| 14   | 0.92      | 0.85   | 0.89     | 41      |
| 15   | 0.99      | 0.99   | 0.99     | 185     |
| 16   | 0.91      | 0.88   | 0.9      | 877     |
| 17   | 1         | 0.88   | 0.93     | 16      |
| 18   | 0.91      | 0.91   | 0.91     | 332     |
| 19   | 0.8       | 0.8    | 0.8      | 106     |
| 20   | 0.5       | 0.36   | 0.42     | 11      |
| 21   | 0.97      | 0.97   | 0.97     | 1225    |
| 22   | 0.93      | 0.93   | 0.93     | 1210    |
| 23   | 0.98      | 0.99   | 0.99     | 1995    |
| 24   | 0.44      | 0.4    | 0.42     | 10      |
| 25   | 1         | 1      | 1        | 139     |
| 26   | 0.97      | 0.98   | 0.97     | 1104    |
| 27   | 0.66      | 0.65   | 0.66     | 415     |
| 28   | 1         | 1      | 1        | 1985    |
| 29   | 0.97      | 0.99   | 0.98     | 1295    |
| 30   | 1         | 1      | 1        | 5       |
| 31   | 0.99      | 0.98   | 0.99     | 197     |
| 32   | 0.87      | 0.8    | 0.83     | 199     |
| 33   | 0.97      | 0.96   | 0.97     | 7285    |
| 34   | 0.99      | 0.99   | 0.99     | 4207    |
| 35   | 0.38      | 0.33   | 0.35     | 9       |
| 36   | 0.96      | 0.98   | 0.97     | 135     |
| 37   | 0.92      | 0.89   | 0.9      | 169     |
| 38   | 1         | 0.67   | 0.8      | 3       |
| 39   | 0.5       | 0.33   | 0.4      | 3       |
| 40   | 0.84      | 0.77   | 0.81     | 483     |
| 41   | 0.92      | 0.9    | 0.91     | 751     |
| 42   | 0.97      | 0.98   | 0.97     | 397     |
| 43   | 0.96      | 0.96   | 0.96     | 1702    |
| 44   | 1         | 0.33   | 0.5      | 3       |
| 45   | 0.69      | 0.82   | 0.75     | 79      |
| 46   | 0.88      | 0.87   | 0.87     | 3373    |
| 47   | 0.96      | 0.95   | 0.95     | 26826   |
| 48   | 0         | 0      | 0        | 1       |
| 49   | 0.89      | 0.92   | 0.9      | 8681    |
| 50   | 0.83      | 0.75   | 0.79     | 32      |
| 51   | 0.83      | 0.85   | 0.84     | 486     |
| 52   | 0.93      | 0.97   | 0.95     | 29      |
| 53   | 0.93      | 0.79   | 0.85     | 48      |
| 54   | 0.72      | 0.54   | 0.62     | 72      |
| 55   | 1         | 1      | 1        | 2       |
| 56   | 0.89      | 0.84   | 0.86     | 49      |
| 57   | 1         | 1      | 1        | 1421    |
| 58   | 0.92      | 0.92   | 0.92     | 89      |
| 59   | 0.81      | 0.85   | 0.83     | 621     |
| 60   | 0.98      | 0.98   | 0.98     | 46622   |
| 61   | 0.94      | 0.94   | 0.94     | 6891    |
| 62   | 1         | 0.99   | 0.99     | 404     |
| 63   | 0.98      | 0.98   | 0.98     | 732     |
| 64   | 0.99      | 0.99   | 0.99     | 7377    |
| 65   | 0.99      | 0.99   | 0.99     | 1676    |
| 66   | 0.94      | 0.98   | 0.96     | 169     |
| 67   | 0.99      | 0.99   | 0.99     | 913     |
| 68   | 0.99      | 0.99   | 0.99     | 1680    |
| 69   | 0.92      | 0.81   | 0.86     | 160     |
| 70   | 0.99      | 0.99   | 0.99     | 213     |
| 71   | 0.98      | 0.99   | 0.98     | 357     |
| 72   | 1         | 0.33   | 0.5      | 3       |
| 73   | 0.99      | 0.98   | 0.99     | 733     |
| 74   | 0.99      | 1      | 0.99     | 684     |
| 75   | 0.59      | 0.48   | 0.53     | 164     |
| 76   | 0.94      | 0.97   | 0.96     | 157     |
| 77   | 0.97      | 0.96   | 0.97     | 1782    |
| 78   | 1         | 1      | 1        | 1418    |
| 79   | 0.92      | 0.93   | 0.93     | 102     |
| 80   | 0.92      | 0.84   | 0.88     | 402     |
| 81   | 0.88      | 0.93   | 0.9      | 40      |
| 82   | 0.99      | 1      | 1        | 2484    |
| 83   | 1         | 1      | 1        | 460     |
| 84   | 0.74      | 0.52   | 0.61     | 27      |
| 85   | 0.91      | 0.93   | 0.92     | 44      |
| 86   | 0.97      | 0.97   | 0.97     | 662     |
| 87   | 0.99      | 0.99   | 0.99     | 3057    |
| 88   | 1         | 1      | 1        | 679     |
| 89   | 0.71      | 0.71   | 0.71     | 406     |
| 90   | 0.98      | 0.98   | 0.98     | 770     |
| 91   | 0.99      | 0.99   | 0.99     | 2934    |
| 92   | 0.78      | 0.82   | 0.8      | 710     |
| 93   | 0.93      | 0.93   | 0.93     | 41      |
| 94   | 1         | 1      | 1        | 538     |
| 95   | 1         | 1      | 1        | 28      |
| 96   | 0.97      | 0.94   | 0.96     | 868     |
| 97   | 1         | 1      | 1        | 14776   |
| 98   | 0.99      | 0.99   | 0.99     | 5514    |
| 99   | 1         | 1      | 1        | 7205    |
| 100  | 0.99      | 0.99   | 0.99     | 3413    |
| 101  | 0.98      | 0.98   | 0.98     | 8140    |
| 102  | 0.9       | 0.81   | 0.85     | 372     |

### References

[1] Y. Cui, W. Che, T. Liu, B. Qin, Z. Yang, S. Wang, and G. Hu.  Pre-trainingwith whole word masking for chinese BERT.CoRR, abs/1906.08101, 2019. URL http://arxiv.org/abs/1906.08101.

[2] J. Devlin, M. Chang, K. Lee, and K. Toutanova.  BERT: pre-training of deepbidirectional transformers for language understanding. In J. Burstein, C. Doran,and T. Solorio, editors,Proceedings of the 2019 Conference of the North Ameri-can Chapter of the Association for Computational Linguistics: Human Language2
Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume1 (Long and Short Papers), pages 4171–4186. Association for Computational Lin-guistics, 2019. doi: 10.18653/v1/n19-1423. URL https://doi.org/10.18653/v1/n19-1423.

[3] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettle-moyer, and V. Stoyanov.  Roberta: A robustly optimized BERT pretraining ap-proach.CoRR, abs/1907.11692, 2019. URL http://arxiv.org/abs/1907.11692.

[4]A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser,and I. Polosukhin.  Attention is all you need.  In I. Guyon, U. von Luxburg,S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett,editors,Advances in Neural Information Processing Systems 30: Annual Confer-ence on Neural Information Processing Systems 2017, December 4-9, 2017, LongBeach, CA, USA, pages 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.