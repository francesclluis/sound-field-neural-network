config.json - Configuring a session
----

The parameters present in a `config.json` file allow one to configure a session. Each of these parameters is described below:

### Dataset
Where to find and how to handle the data.
* **path**: *string* Path to dataset
* **factor**: *int* Value of the factor applied during downsampling and upsampling
* **xSamples**: *int* Number of ground truth samples along the x-axis in the room. (32 if using the provided dataset)
* **ySamples**: *int* Number of ground truth samples along the y-axis in the room. (32 if using the provided dataset)

### Training
How training will be performed.
* **batch_size**: *int* Number of samples in a batch
* **num_epochs**: *int* Number of epochs to train for
* **num_steps_val**: *int* Total number of steps (batches of samples) to yield from validation generator before stopping at the end of every epoch.
* **num_steps_train**: *int* Total number of steps (batches of samples) to yield from validation generator before stopping at the end of every epoch.
* **session_id**: *int* Numerical identifier for this session
* **lr**: *float* Learning rate
* **loss**: *float* 
  * **valid_weight**: *float* Weight given to the loss term considering microphone position predictions.
  * **hole_weight**: *float* Weight given to the loss term considering non-microphone position predictions.
 
### Evaluation
How evaluation will be performed.
* **min_mics**: *int* Minimum number of microphones placed in a room to evaluate the model.
* **max_mics**: *int* Maximum+1 number of microphones placed in a room to evaluate the model.
* **step_mics**: *int* Spacing between the value of the number of microphones placed.
* **num_comb**: *int* Number of different irregular patterns tested with a fix amount of microphones.

### Visualization
How visualization will be performed.
* **num_mics**: *int* Number of microphones randomly located in the real room.
* **source**: *int* Numerical identifier of the source location. Must be either 0 or 1.
