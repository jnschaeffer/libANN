#include "ANN_ffnetwork.h"

void FFN_Close(ANN_FFNetwork *n) {
  if(n != NULL) {
    if(n->dumpFile != NULL) {
      free(n->dumpFile);
    }

    if(n->hiddenLayers != NULL) {
      free(n->hiddenLayers);
    }

    free(n);
  }
}

/** 
 * \fn ANN_FFNetwork *FFN_Create(int numInputs, int numOutputs, 
 * int numHiddenLayers, int hiddenLayerSize, int maxEpochs,
 * double learningRate, double minError, double bias,
 * double momentum)
 *
 * \brief Builds a feed-forward neural network with the specified parameters
 *
 * \param numInputs Number of inputs for the network
 * \param numOutputs Number of outputs for the network
 * \param numHiddenLayers Number of hidden layers for the network
 * \param hiddenLayerSize Size of each hidden layer
 * \param maxEpochs Maximum number of training epochs allowed when training
 * \param learningRate Speed at which the network will learn
 * \param minError Minimum acceptable error
 * \param bias Bias weight for each node
 * \param momentum Momentum of the network
 *
 * \return Pointer to the feed-forward neural network
 */
ANN_FFNetwork *FFN_Create(
  int numInputs, int numOutputs, int numHiddenLayers, 
  int hiddenLayerSize, int maxEpochs,double learningRate, 
  double minError, double bias, double momentum) {
 
  //First thing's first...seed the RNG
  srand(time(NULL));

  int i = 0;
  ANN_FFNetwork *n = malloc(sizeof(ANN_FFNetwork));

  //Out of memory
  if(n == NULL) {
    return NULL;
  }

  n->dumpFile = NULL;
  n->dumpFilename = NULL;
  n->numInputs = numInputs;
  n->numOutputs = numOutputs;
  n->numHiddenLayers = numHiddenLayers;
  n->hiddenLayerSize = hiddenLayerSize;
  n->maxEpochs = maxEpochs;
  n->learningRate = learningRate;
  n->minError = minError;
  n->bias = bias;
  n->momentum = momentum;

  n->hiddenLayers = malloc(sizeof(ANN_Layer) * n->numHiddenLayers);
 
  //Out of memory 
  if(n->hiddenLayers == NULL) {
    free(n);
    return NULL;
  }

  //Initialize the input layer
  DEBUG_FFNET(("Initializing input layer...\n"));
  LAYER_Init(&n->inputLayer,n->numInputs,n->bias,n->learningRate,n->momentum,NULL);
  DEBUG_FFNET(("...done.\n\n"));

  //Initialize the first input layer
  DEBUG_FFNET(("Initializing hidden layer 0...\n"));
  LAYER_Init(&(n->hiddenLayers[0]),n->numInputs,n->bias,n->learningRate,n->momentum,&(n->inputLayer));
  DEBUG_FFNET(("...done.\n\n"));

  //Initialize the other hidden layers (if there are any)
  for(i = 1; i < n->numHiddenLayers; i++) {
    DEBUG_FFNET(("Initializing hidden layer %d...\n",i));
    LAYER_Init(&n->hiddenLayers[i],n->hiddenLayerSize,n->bias,n->learningRate,n->momentum,&n->hiddenLayers[i-1]);
    DEBUG_FFNET(("...done.\n\n"));
  }

  //Initialize the output layer
  DEBUG_FFNET(("Initializing output layer...\n"));
  LAYER_Init(&n->outputLayer,n->numOutputs,n->bias,n->learningRate,n->momentum,&n->hiddenLayers[n->numHiddenLayers - 1]);
  DEBUG_FFNET(("...done.\n\n"));

  //Set the input layer's output layer to be the first hidden layer
  n->inputLayer.outputLayer = &n->hiddenLayers[0];

  //Set the hidden layers' output layers to be the next hidden layers
  for(i = 0; i < n->numHiddenLayers - 1; i++) {
    n->hiddenLayers[i].outputLayer = &n->hiddenLayers[i+1];
  }

  //Set the final hidden layer's output layer to be the output layer
  n->hiddenLayers[n->numHiddenLayers - 1].outputLayer = &n->outputLayer;

  return n;
}

/**
 * \fn int FFN_SetInputs(ANN_FFNetwork *n, double *values)
 *
 * Sets the inputs for the neural network.
 *
 * \param n Pointer to an ANN_FFNetwork
 * \param values Array of doubles (should be the same size as n->numInputs)
 *
 * \return 0 if values exists, -1 if it does not
 */
int FFN_SetInputs(ANN_FFNetwork *n, double *values) {
  //Return 0 if values exists
  if(values != NULL) {
    n->inputLayer.outputValues = values;
    return 0;
  } else {
    //Otherwise, return -1
    return -1;
  }
}

/**
 * \fn int FFN_SetDesiredOutputs(ANN_FFNetwork *n, double *values)
 *
 * Sets the desired outputs for the output layer to values.
 *
 * \param n Pointer to an ANN_FFNetwork
 * \param values Array of doubles (should be the same size as n->numOutputs)
 *
 * \return 0 if values exists, -1 if it does not
 */
int FFN_SetDesiredOutputs(ANN_FFNetwork *n, double *values) {
  //Return 0 if values exists
  if(values != NULL) {
    n->outputLayer.desiredValues = values;
    return 0;
  } else {
    //Otherwise, return -1
    return -1;
  }
}

/**
 * \fn void FFN_FeedForward(ANN_FFNetwork *n)
 *
 * Feeds all data through the input layer, hidden layer(s), and then output layer.
 * 
 * \param n A pointer to an ANN_FFNetwork
 * \return The values that the network has set as the output
 */
double *FFN_FeedForward(ANN_FFNetwork *n) {
  int i = 0;

  //DEBUG: Print all of the input values
  DEBUG_FFNET(("Input: "));
  for(i = 0; i < n->numInputs; i++) {
    DEBUG_FFNET(("%.3f ",n->inputLayer.outputValues[i]));
  }
  DEBUG_FFNET(("\n"));

  //Feed the input values through the hidden layer(s)
  for(i = 0; i < n->numHiddenLayers; i++) {
    LAYER_FeedForward(&n->hiddenLayers[i]);
  }
  LAYER_FeedForward(&n->outputLayer);
  DEBUG_FFNET(("Output: "));
  for(i = 0; i < n->numOutputs; i++) {
    DEBUG_FFNET(("%.3f ",n->outputLayer.outputValues[i]));
  }
  DEBUG_FFNET(("\n"));

  return n->outputLayer.outputValues;
}

/**
 * Returns the output values for the network.
 *
 * \param n Pointer to an ANN_FFNetwork
 */
double *FFN_GetOutputs(ANN_FFNetwork *n) {
  return n->outputLayer.outputValues;
}

/**
 * Calculates the errors through the entire network.
 *
 * \param n Pointer to an ANN_FFNetwork
 */
void FFN_CalcErrors(ANN_FFNetwork *n) {
  int i = 0;

  DEBUG_FFNET(("Desired values: "));
  for(i = 0; i < n->numOutputs; i++) {
    DEBUG_FFNET(("%.3f ",n->outputLayer.desiredValues[i]));
    LAYER_CalcErrors(&n->outputLayer);
  }
  DEBUG_FFNET(("\n"));
  LAYER_CalcErrors(&n->hiddenLayers[i]);
  for(i = n->numHiddenLayers - 1; i >= 0; i--) {
    LAYER_CalcErrors(&n->hiddenLayers[i]);
  }
}

/**
 * Sums up all of the errors in the network's output layer.
 *
 * \param n Pointer to an ANN_FFNetwork
 *
 * \return The sum of outputLayer.errors[]
 */
double FFN_SumErrors(ANN_FFNetwork *n) {
  int i = 0;
  double sum = 0.0;

  for(i = 0; i < n->numOutputs; i++) {
    sum += n->outputLayer.errors[i];
  }

  return sum;
}

/**
 * \brief Adjusts weights according to errors from the previous epoch.
 *
 * NOTE: To avoid errors, this function should be run after all others in the epoch.
 *
 * \param n Pointer to an ANN_FFNetwork
 */
void FFN_AdjustWeights(ANN_FFNetwork *n) {
  int i = 0;

  LAYER_AdjustWeights(&(n->outputLayer));
  for(i = n->numHiddenLayers - 1; i >= 0; i--) {
    LAYER_AdjustWeights(&(n->hiddenLayers[i]));
  }
}

/**
 * Trains the neural network using the first numSets of the training data until maxEpochs is reached or minError is reached
 * 
 * \param n Pointer to an ANN_FFNetwork
 * \param numSets Number of sets of data to be used
 * \param trainingInputs Matrix of all training inputs
 * \param trainingOutputs Matrix of all desired outputs
 */
void FFN_Train(ANN_FFNetwork *n, int numSets, double **trainingInputs, double **trainingOutputs) {
  int i,j;
  i = j = 0;

  n->trainingInputs = trainingInputs;
  n->trainingOutputs = trainingOutputs;
  int epochs = 0;
  double avgErr;

  do {
    double avgErr = 0.0;
    for(i = 0; i < numSets; i++) {
      n->inputLayer.outputValues = n->trainingInputs[i];
      n->outputLayer.desiredValues = n->trainingOutputs[i];

      FFN_FeedForward(n);
      FFN_CalcErrors(n);
      FFN_AdjustWeights(n);

      double sum = FFN_SumErrors(n);
      avgErr += (sum * sum) / n->numOutputs;
      DEBUG_FFNET(("EPOCH %d OF %d:\n", epochs, n->maxEpochs));
      DEBUG_FFNET(("Output should be %.3f and is %.3f\n",n->trainingOutputs[i][0],n->outputLayer.outputValues[0]));
    }
    avgErr /= numSets;
    DEBUG_FFNET(("Average error: %.3f\n",avgErr));

    epochs++;
  } while(avgErr < n->minError && epochs < n->maxEpochs);
}
