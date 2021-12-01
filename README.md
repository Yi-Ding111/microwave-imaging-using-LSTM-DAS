# stroke microwave imaging algorithm

This algorithm is using signal data (snp file) to do imaging. It is divided into two parts:

__boundary reconstruction__

__stroke position reconstruction__

All the code can be find in __reconstruction model__ file.

> ## boundary reconstruction

This part reconstruction uses signal data (focus on boundary detection) to try to build up brain boundary. It is implemented mainly through following scripts:

* antenna_boundary_position.py

* signal_porcess.py

* data_prepare.py

* learning_model.py

* boundary_reconstruction.py

This algorithm includes LSTM, Bezier curve, IFFT transform, signal data processing and etc. 

> ## stroke position reconstruction

This part reconstruction uses signal data (focus on stroke detection) to try to localize the stroke position. It is implemented mainly through following scripts:

* target_data_pre.py

* beamformer.py

* signal_process.py

* target_find.py

* target_reconstruction.py

This algorithm includes delay-and-sum, k-means, direction of arrival, signal data processing and etc. 

___
___

### Boundary reconstruction output:

```
boundary_reconstruction.py
boundary_reconstruction.ipynb
```

### stroke position reconstruction output:

``` 
target_reconstruction.py
target_reconstruction.ipynb
```

___
___

## lstm model

Lstm trained models are stored in *trainingmodel* file. 


## Contact

Author: Yi Ding
Supervisor: Dr. Zhu
email: dydifferent@gmail.com
