#include "ANN_layer.h"
/*
NeuronLayer(void) {
  srand(time(NULL));
  n->inputLayer = NULL;
  n->outputLayer = NULL;

  n->inputSums = NULL;
  n->outputValues = NULL;
  n->desiredValues = NULL;
  n->errors = NULL;

  n->inputWeightMatrix = NULL;
  n->biasWeights = NULL;
}
*/

void LAYER_FreeLayer(ANN_Layer *n) {
  int i,j;
  i = j = 0;

  n->inputLayer = n->outputLayer = NULL;
  if(n->inputSums != NULL) {
    free(n->inputSums);
  }
  if(n->outputValues != NULL) {
    free(n->outputValues);
  }

  if(n->desiredValues != NULL) {
    free(n->desiredValues);
  }

  if(n->errors != NULL) {
    free(n->errors);
  }

  if(n->biasWeights != NULL) {
    free(n->biasWeights);
  }

  if(n->inputWeightMatrix != NULL) {
    for(i = 0; i < n->size; i++) {
      free(n->inputWeightMatrix[i]);
    }
    free(n->inputWeightMatrix);
  }
}

double LAYER_RandDouble() {
  double num = (double)(rand());
  double dem = (double)(RAND_MAX);
  return num / dem;
}

double LAYER_Sgm(double x) {
  return 1.0/(1.0 + exp(-x));
}

//initialize neuron layer
void LAYER_Init(
  ANN_Layer *n, int size_, double bias_, 
  double learningRate_, double momentum_, ANN_Layer *inputLayer_) {

  DEBUG_LAYER(("Beginning layer creation.\n"));
  int i,j;
  i = j = 0;

  DEBUG_LAYER(("Assigning parameters..."));

  n->size = size_;
  n->bias = bias_;
  n->learningRate = learningRate_;
  n->momentum = momentum_;
  n->inputLayer = inputLayer_;
  DEBUG_LAYER(("done.\n"));

  DEBUG_LAYER(("Allocating memory..."));
  n->outputValues = malloc(sizeof(double) * n->size); //new double[size];
  DEBUG_LAYER(("1,"));
  n->desiredValues = malloc(sizeof(double) * n->size); //new double[size];
  DEBUG_LAYER(("2,"));
  n->errors = malloc(sizeof(double) * n->size); //new double[size];
  DEBUG_LAYER(("done.\n"));
  for(i = 0; i < n->size; i++) {
    n->outputValues[i] = n->desiredValues[i] = n->errors[i] = 0.0;
  }

  //do this if the layer has an input layer
  if(n->inputLayer != NULL) {
    //allocate memory for nodes
    n->numInputs = n->inputLayer->size;

    n->biasWeights = malloc(sizeof(double) * n->size);

    n->inputWeightMatrix = malloc(sizeof(double*) * n->size);
    for(i = 0; i < n->size; i++) {
      n->inputWeightMatrix[i] = malloc(sizeof(double) * n->numInputs);
    }

    n->weightChanges = malloc(sizeof(double*) * n->size);
    for(i = 0; i < n->size; i++) {
      n->weightChanges[i] = malloc(sizeof(double) * n->numInputs);
    }

    n->inputSums = malloc(sizeof(double) * n->size);

    //randomize weights
    for(i = 0; i < n->size; i++) {
      n->biasWeights[i] = LAYER_RandDouble();
    }

    for(i = 0; i < n->size; i++) {
      for(j = 0; j < n->numInputs; j++) {
        n->inputWeightMatrix[i][j] = LAYER_RandDouble();
      }
    }

    for(i = 0; i < n->size; i++) {
      for(j = 0; j < n->numInputs; j++) {
        n->weightChanges[i][j] = 0.0;
      }
    }

    for(i = 0; i < n->size; i++) {
      n->inputSums[i] = 0.0;
    }
  }
}

//process inputs, set outputs
void LAYER_FeedForward(ANN_Layer *n) {
  DEBUG_LAYER(("Processing inputs..."));
  int i,j;
  i = j = 0;
  for(i = 0; i < n->size; i++) {
    //DEBUG_LAYER(("  %d:\n",i);
    n->inputSums[i] = 0.0;
    for(j = 0; j < n->numInputs; j++) {
      //DEBUG_LAYER(("    %.3f * %.3f\n",n->inputWeightMatrix[i][j],inputLayer->outputValues[j]);
      n->inputSums[i] += n->inputWeightMatrix[i][j] * n->inputLayer->outputValues[j];
    }
    n->inputSums[i] += n->biasWeights[i] * n->bias;

    n->outputValues[i] = LAYER_Sgm(n->inputSums[i]);
    //DEBUG_LAYER(("    Output = %1.3f\n",outputValues[i]);
  }
  DEBUG_LAYER(("Done.\n"));
}

void LAYER_CalcErrors(ANN_Layer *n) {
  int i,j;
  i = j = 0;

  DEBUG_LAYER(("Calculating errors..."));
  if(n->inputLayer == NULL) {
    DEBUG_LAYER(("input layer has no errors.\n"));
    return;
  }
  //DEBUG_LAYER(("\n");
  if(n->outputLayer != NULL) {
    double errorSum;
    for(i = 0; i < n->size; i++) {
      errorSum = 0.0;
      DEBUG_LAYER(("  "));
      for(j = 0; j < n->outputLayer->size; j++) {
        DEBUG_LAYER(("%.3f ",n->outputLayer->inputWeightMatrix[j][i] * n->outputLayer->errors[j]));
        errorSum += n->outputLayer->inputWeightMatrix[j][i] * n->outputLayer->errors[j];
      }
      n->errors[i] = errorSum * n->outputValues[i] * (1 - n->outputValues[i]);
      DEBUG_LAYER(("%.3f ",n->errors[i]));
    }
  } else { //if not, this is the output layer
    for(i = 0; i < n->size; i++) {
      n->errors[i] = (n->desiredValues[i] - n->outputValues[i]) * n->outputValues[i] * (1.0 - n->outputValues[i]);
      DEBUG_LAYER(("%.3f ",n->errors[i]));
    }
  }
  DEBUG_LAYER(("...done.\n"));
}

void LAYER_AdjustWeights(ANN_Layer *n) {
  int i,j;
  i = j = 0;

  DEBUG_LAYER(("Adjusting weights...\n"));
  //if this is not the input layer...
  if(n->inputLayer != NULL) {
    for(i = 0; i < n->size; i++) {
      for(j = 0; j < n->numInputs; j++) {
        //multiply the above layer's errors by the current neuron's output value
        double changeAmt = n->learningRate * n->errors[i] * n->inputLayer->outputValues[j];
        DEBUG_LAYER(("  %.3f + %.3f + %.3f = ",n->inputWeightMatrix[i][j],(n->momentum * n->weightChanges[i][j]),changeAmt));
        n->inputWeightMatrix[i][j] += (n->learningRate * n->errors[i] * n->inputLayer->outputValues[j]) + (n->momentum * n->weightChanges[i][j]);

        n->weightChanges[i][j] = changeAmt;
        DEBUG_LAYER(("%.3f\n", n->errors[i] * n->inputLayer->outputValues[j]));
      }
      n->biasWeights[i] += n->errors[i] * n->bias;
      DEBUG_LAYER(("\n"));
    }
  }
  DEBUG_LAYER(("...done.\n\n"));
}
