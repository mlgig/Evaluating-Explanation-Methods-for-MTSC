# Evaluating-Explanation-Methods-for-MTSC

Accepted for AALTD 2023 (8th International Workshop on Advanced Analytics and Learning on Temporal Data, ECMLPKDD 2023)

## Abstract:

Multivariate time series classification is an important computational task arising in applications where data is recorded over time and over multiple channels. For example, a smartwatch can record the acceleration and orientation of a person's motion, and these signals are recorded as multivariate time series. We can classify this data to understand and predict human movement and various properties such as fitness levels. In many applications classification alone is not enough, we often need to classify but also understand what the model learns (e.g., why was a prediction given, based on what information in the data). The main focus of this paper is on analysing and evaluating explanation methods tailored to Multivariate Time Series Classification (MTSC). We focus on saliency-based explanation methods that can point out the most relevant channels and time series points for the classification decision. We analyse two popular and accurate multivariate time series classifiers, ROCKET and dResNet, as well as two popular explanation methods, SHAP and dCAM. We study these methods on 3 synthetic datasets and 2 real-world datasets and provide a quantitative and qualitative analysis of the explanations provided. We find that flattening the multivariate datasets by concatenating the channels works as well as using multivariate classifiers directly and adaptations of SHAP for MTSC work quite well. Additionally, we also find that the popular synthetic datasets we used are not suitable for time series analysis. 

## Results:

![image](https://github.com/mlgig/Evaluating-Explanation-Methods-for-MTSC/blob/main/imgs/accuracy.png) 


![image](https://github.com/mlgig/Evaluating-Explanation-Methods-for-MTSC/blob/main/imgs/synthetic_results.png) 


![image](https://github.com/mlgig/Evaluating-Explanation-Methods-for-MTSC/blob/main/imgs/realWorld_results.png) 

## How to cite:

```
@misc{serramazza2023evaluating,
      title={Evaluating Explanation Methods for Multivariate Time Series Classification}, 
      author={Davide Italo Serramazza and Thu Trang Nguyen and Thach Le Nguyen and Georgiana Ifrim},
      year={2023},
      eprint={2308.15223},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
