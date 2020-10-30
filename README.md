# Combining Active Learning with Self-training Scheme

This repository contains a Python implementation of a combined framework (_AL+SSL_) that exploits both Active and Semi-supervised Learning concepts over structured data. Our utmost aim is to alleviate the human effort during the learning stage from unlabeled data. Therefore, we exploit the aforemntioned concepts through appropriate learning startegies that operate under a complementary manner for detecting informative unlabaled instances and highly affect the decision function of each exploited probabilistic learner. Furthermore, by inserting an additional stage that rejects candidate incoming unlabeled instances despite the initial confidence of SSL concept, we manage to reduce noisy decisions.

This work constitutes an extended version of the originally proposed conference [paper](https://ieeexplore.ieee.org/document/8900724) published in [IISA2019](http://iisa2019.upatras.gr/) conference. Now, it accompanies the article with title **'Classification of acoustical signals by combining Active Learning strategies with Semi-supervised Learning schemes'**  that has been submitted to [NCAA](https://link.springer.com/journal/521) journal.

As it concerns the specific strategies of AL and SSL concepts that are employed, it combines Uncertainty Sampling Query Strategy along with Self-training schemes so as to obtain both effective and efficinet approaches. Eight different learners were examined into the proposed framework tackling with problems that stem from acoustical signals, presenting both quintitative and qualitative results regarding their performance.

We provide here some initial information about the hyperparameters of the combined _AL+SSL_ scheme, as well as the manner of inserting any other probabilistic classifier into our script (found in the folder Algorithms):

- By modifying the class _Basemodel(object)_ similarly with our examined learning models (RfModel, ExtModel etc.) any probabilistic classifier can fit to our mechanism. In order to keep up with the rest requirements, we have to add its name as a string as well as the name of the class in the lists models_str and models, respectively,
- *LR* parameter denotes the Labeled Ratio value of our experiment. This is computed by dividing the number of the initially labeled instances with the total amount of both labeled and unlabeled instances.
- *ssl_ration* parameter denotes the participation level of each learning concept between AL and SSL,
- *query_pools* denotes the budget of unlabeled isntances that is permitted to retrieve during our experiment,
- *K* list denotes all the examined sizes of batches that are investigated for consuming the provided budget.


After the final decision of journal's reviewers, more insights are going to be provided regarding the examined datasets, the parameters of the proposed scheme and the use of the included tools/scripts for testing or expanding further our work.

The rest of authors who contributed to this repository are Chris Aridas, Dr. Vasileios G. Kanas, and our supervisor Dr. Sotiris Kotsiantis.

Our academic site: http://ml.math.upatras.gr/.
