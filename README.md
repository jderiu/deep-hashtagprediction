# deep-hashtagprediction
Deep Convolutional Neural Network for Twitter Hashtag Prediction

## Preparation ##
To prepare the project for execution you need to do the following:

 - Put your tweets inside the tweets/ folder. Note that they should be gzipped. They should provide a test and a train file. Adapt the parse_tweets.py script.
 - If you already have word embeddings place them inside the embeddings/ folder and adapt the extract_embeddings.py. You need to change the wemb_ variables and define the name of the file, the delimiter used and the number of dimensions.
 
## Preprocessing ##
 - If you don't have word embeddings you may run the 'create_wordembeddings.py' script. In the main method declare the files with the tweets you are using. In current case only the tweets/hastag_tweets.gz are used. The output is stored in embeddings/hashtag_tweets_embedding.
 - Parse tweets: Run the 'parse_tweets.py' script. It uses as input the files with the training and test data. It creates 6 files. It creates the tweet-representation for both the training and test data. It creates the hashtag-representation for both the training and the hashtag. It creates 2 index, one for the twitter words and one for the hashtags.
 - Extract_embeddings: it creates 2 lookup tables with word-idx -> word-embedding. One lookup table for the twitter-words and one for the hashtags. Run the 'extract_embeddings.py' script.
 
## Training ##
 Run the training.py script as follows:
 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python training.py

You need to have CUDA and CuDNN installed.

## Prepare Amazon EC2 ##

 - Download and install Anaconda 2.7: https://www.continuum.io/downloads
 -  Install Theano:
	 - sudo apt-get update
	 - sudo apt-get -y dist-upgrade-
	 - screen -S theano (this just opens a new terminal)
	 - sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev (installs all the g++ and blas dependencies)
	 - pip install theano
 - Install CUDA:
	 - sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
	 - sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
	 - sudo apt-get update
	 - sudo apt-get install -y cuda
	 - echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
	 - echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc
 - sudo reboot
 - Install CuDNN
	 - Download cudnn: https://developer.nvidia.com/cudnn
	 - tar -zxf cudnn-7.0-linux-x64-v3.0-prod.tgz
	 - cd cuda
	 - sudo cp lib64/* /usr/local/cuda/lib64/
	 - sudo cp include/cudnn.h /usr/local/cuda/include/
	 