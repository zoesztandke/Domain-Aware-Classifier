# Domain-Aware Classification 
This project explores how to build a classification system that can recognise whether an input sample comes from the same distribution as the labelled training data. The model is trained using both in-domain and out-domain datasets: the in-domain data is used to learn the actual classification task, and both datasets are used to train a separate domain classifier. During evaluation, the system should perform well on in-domain inputs and intentionally perform poorly on out-domain inputs. 

## 1. Problem Formulation
This assignment focuses on understanding model behaviour using four datasets:
- in-domain-train (labelled)
- out-domain-train (unlabelled)
- in-domain-eval (labelled)
- out-domain-eval (labelled)

The goal is to build a model that performs well on in-domain data and deliberately performs poorly on out-domain data. This setup highlights how explicit domain detection can control model behaviour under mismatched distributions. 

## 2. Methodology
The system combines two components: 
1. Domain Classifier
A binary classifier trained to distinguished in-domain vs. out-domain samples 		using both training datasets.
2. In-Domain Task Classifier
A multi-class model trained only on the labelled in-domain training data.

At inference time, the domain classifier decides whether a sample should be handled by the real classifier or intentionally degraded. 

During evaluation:
1. The domain classifier predicts whether the input is in-domain.
2. If in-domain, the in-domain classifier produces the final prediction.
3. If out-domain, the system outputs a random class, ensuring intentionally low performance on out-domain evaluation samples. 

## 3. Training Constraints and Current Limitations
Training must fit within a 30-minute runtime limit, which restricts training time and model size. Because of this constraint, current accuracies are lower. Further training outside the time limit is expected to improve performance. 

## 4. Experimental Results
1e-4
epochs 60/60
- Domain Classifier Accuracy: 65.0%
- Object Classifier Accuracy: 44%
- In-Domain Classification Accuracy: 41.3%
- Out-Domain Classification Accuracy: 21.6%

1e-3
epochs 60/60
- Domain Classifier Accuracy: 63.9%
- Object Classifier Accuracy: 52.0%
- In-Domain Classification Accuracy: 41.9%
- Out-Domain Classification Accuracy: 17.8%

1e-2
epochs 45/75
- Domain Classifier Accuracy:
- Object Classifier Accuracy:
- In-Domain Classification Accuracy:
- Out-Domain Classification Accuracy:

These results are achieved through running in Google Colab with T4 GPU in approximately 30 minutes. 






