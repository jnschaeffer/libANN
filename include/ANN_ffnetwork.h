#include "ANN_layer.h"

#ifndef __ANN_FFNETWORK_H
#define __ANN_FFNETWORK_H

#ifdef _DEBUG_FFNET
  #define DEBUG_FFNET(x) printf x
#else
  #define DEBUG_FFNET(x)
#endif

BEGIN_C_DECLS

/// A simple feed-forward neural network.
/**
 * \struct ANN_FFNetwork
 *
 * \brief A simple feed-forward neural network.
 *
 * The ANN_FFNetwork structure can be used on its own (with random weights) and be trained manually, or may be trained by using the FFN_Train() function.
 */
typedef struct ANN_FFNetwork {
  ///The dumpfile.  Not being used yet.
  void *dumpFile;
  
  ///The dumpfile path
  char *dumpFilename; 
  
  ///Neural network learning rate
  double learningRate;

  ///Minimum allowable error
  double minError;

  ///Bias weight assigned to each node
  double bias;

  ///Momentum of network
  double momentum;

  ///Number of inputs
  int numInputs;

  ///Size of hidden layer
  int hiddenLayerSize; 

  ///Number of hidden layers
  int numHiddenLayers;

  ///Number of outputs
  int numOutputs;

  ///Maximum number of training epochs
  int maxEpochs;
  
  ///Input layer of the network
  struct ANN_Layer inputLayer;

  ///Hidden layers of the network
  struct ANN_Layer *hiddenLayers;

  ///Output layer of the network
  struct ANN_Layer outputLayer;

  ///Training input matrix
  double **trainingInputs;
  
  ///Training output matrix
  double **trainingOutputs;
} ANN_FFNetwork;

/* Functions */

/// \brief Creates the network
ANN_FFNetwork *FFN_Create(int,int,int,int,int,double,double,double,double);

/// \brief Closes the network
void FFN_Close(ANN_FFNetwork*);

/// \brief Sets inputs for the network
int FFN_SetInputs(ANN_FFNetwork*,double*);

/// \brief Sets desired values for the network
int FFN_SetDesiredOutputs(ANN_FFNetwork*,double*);

/// \brief Processes the inputs layer-by-layer
double *FFN_FeedForward(ANN_FFNetwork*);

/// \brief Returns the output values of the network
double *FFN_GetOutputs(ANN_FFNetwork*);

// \brief Calculates the error between desired output and real output
//
//NOTE: outputLayer.desiredValues MUST NOT BE NULL
void FFN_CalcErrors(ANN_FFNetwork*);

// \brief Sums the neural network's errors
double FFN_SumErrors(ANN_FFNetwork*);

// \brief Adjusts weights according to the summed errors
void FFN_AdjustWeights(ANN_FFNetwork*);

// \brief Trains the network
void FFN_Train(ANN_FFNetwork*,int,double**,double**);

// \brief Dumps the network data into the dumpfile (not yet implemented)
void FFN_DumpNet(ANN_FFNetwork*,char*);

END_C_DECLS

#endif
