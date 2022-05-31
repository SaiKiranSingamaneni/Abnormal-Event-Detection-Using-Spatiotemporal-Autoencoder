# Abnormal-Event-Detection-Using-Spatiotemporal-Autoencoder
Using spatio-temporal data, we created a model that can detect anomalous events in surveillance videos. This project is based on the idea proposed in [Abnormal Event Detection in Videos using Spatiotemporal Autoencoder](https://arxiv.org/pdf/1701.01546.pdf). Everything you need to learn about spatiotemporal autoencoders, model we've created can be read in the aforementioned paper.

# Implementing this project
1. To implement this project, you either need to have better GPU for faster running time or it can be run on any CPU.
2. Install requirements.txt. Try to download in virtual environment. Or download the libraries manually based on your requirements.
3. Download Avenue or UCSD datasets. Put the paths of the folders in the code.
4. Try to run the codes in this order, processor.py -> train.py -> test.py -> start_live_feed.py.
5. Or you can directly run the test.py or start_live_feed.py using pre-trained models.

## Link to the datasets
1. [Avenue Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
2. [UCSD Pedestrian Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
