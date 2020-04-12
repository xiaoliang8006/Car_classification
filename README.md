**This is a project about car classification.**

## Data description

### 4 Classes about car's condition:

	unacc: unaccepted condition
	acc: accepted condition
	good: good condition
	vgood: very good condition
	
### Features:

	buying (购买价: vhigh, high, med, low)
	maint (维护价: vhigh, high, med, low)
	doors (几个门: 2, 3, 4, 5more)
	persons (载人量: 2, 4, more)
	lug_boot (贮存空间: small, med, big)
	safety (安全性: low, med, high)

## Environment

	numpy==1.17.1
	pandas==0.25.0
	tensorflow==1.1.0
	Keras==2.0.4
	torch==0.4.1
	xgboost==0.90


## Training

### Files:

`data_processing.py` : preprocess data, use one-hot embedding and label-encoder.

`model_tensorflow.py` : Use tensorflow framework.

`model_keras.py` : Use keras framework.

`model_pytorch.py` : Use pytorch framework.

`model_xgboost.py` : Implementation with traditional machine learning model XGBoost.
