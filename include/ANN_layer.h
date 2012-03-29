#include "ANN_main.h"
#ifndef __ANN_LAYER_H__
#define __ANN_LAYER_H__

#ifdef _DEBUG_LAYER
  #define DEBUG_LAYER(x) printf x
#else
  #define DEBUG_LAYER(x)
#endif

BEGIN_C_DECLS

 /// A simple neural network layer designed to fit either on its own or in a NeuralNetwork class.
typedef struct ANN_Layer {
  /** \brief Pointer to the node layer "above" this one */
  /// NOTE: Can be NULL, if so then this is assumed to be the input layer
  struct ANN_Layer *inputLayer;

  /** \brief Pointer to the node layer "below" this one */
  /// NOTE: Can be NULL, if so then this is assumed to be the output layer
  struct ANN_Layer *outputLayer;

  /// Layer size
  int size;

  /// Number of inputs in layer (0 if it is the input layer)
  int numInputs;

  /// Bias value of neuron layer
  double bias;

  /// Momentum of layer
  double momentum;

  /// Learning rate of layer
  double learningRate;

  /// Sum of all weighted inputs for each node
  double *inputSums;

  /// Sum of all output values for each node
  double *outputValues;

  /** \brief Sum of all desired values for each node */
  /// NOTE: Can be NULL if network is not being trained automatically
  double *desiredValues;
    
  /// Error values for each node
  double *errors;

  /// Array of input weights for each node
  double **inputWeightMatrix;

  /// Array of weight changes for each node (only used in training)
  double **weightChanges;

  /// Bias weights for each node (weights the bias value)
  double *biasWeights;
} ANN_Layer;
 
/// \brief Initializes the node layer
void LAYER_Init(ANN_Layer*,int,double,double,double,ANN_Layer*);
void LAYER_FreeLayer(ANN_Layer*); 
void LAYER_FeedForward(ANN_Layer*);
void LAYER_CalcErrors(ANN_Layer*);
void LAYER_AdjustWeights(ANN_Layer*);
double LAYER_RandDouble(void);
double LAYER_sgm(double);

END_C_DECLS

#endif
